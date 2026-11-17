"""Finite five-mode archive-adapter checks. Athul Prem | _AP_pnc."""

import copy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from tools import diagnose_conversion_families_AP_pnc as diagnosis


def mode_fixture():
    """Five signed channels whose reverse roots are deliberately nonadjacent."""
    roots = np.array([1, 4, 7, 10, 13])
    reverse = np.array([2, 8, 17, 23, 29])
    waves = np.arange(1, 31).astype(complex)
    waves[roots] = [2, 3, 5, 7, 11]
    waves[reverse] = -waves[roots]
    return {"roots": roots, "reverse_roots": reverse, "k": waves[roots].real, "waves": waves}


def response_fixture(modes=None):
    modes = mode_fixture() if modes is None else modes
    roots, reverse = modes["roots"].tolist(), modes["reverse_roots"].tolist()
    reflection = np.diag(np.array([.1j, .05j, .08j, .12j, .03j]))
    reflection[1, 4], reflection[4, 1] = .03 + .04j, .04 - .03j
    transmission = np.diag(np.sqrt(1 - np.sum(abs(reflection)**2, axis=0)))
    totals, coefficients = [], []
    for column, incoming in enumerate(roots):
        totals.append({"incident_mode_order": column + 1, "frequency_GHz": 1.,
                       "reflection": float(np.sum(abs(reflection[:, column])**2)),
                       "transmission": float(np.sum(abs(transmission[:, column])**2))})
        for kind, outgoing, matrix in (("reflection", reverse, reflection), ("transmission", roots, transmission)):
            for row, root in enumerate(outgoing):
                amplitude = matrix[row, column]
                coefficients.append({"kind": kind, "incident_mode_order": column + 1, "incident_port_index": incoming,
                    "outgoing_port_index": root, "amplitude_real": float(amplitude.real),
                    "amplitude_imag": float(amplitude.imag), "power_fraction": float(abs(amplitude)**2)})
    summary = {"frequency_hz": 1e9, "incident_side": "left", "incident_mode_indices": roots,
               "num_incident_channels": 5, "power_balance_passed": True, "run_hash": "synthetic-run"}
    return summary, totals, coefficients, reflection, transmission


def metadata_fixture():
    control = {"status": "compared", "input_power_checks_passed": True, "power_change_limit": .01,
               "complex_change_limit": .01, "power_passed": True, "complex_passed": False,
               "mesh_pair_passed": False, "maximum_complex_change": .01670625750875372,
               "maximum_power_change": .0000647272965669651, "matched_channels": {}}
    documents = {"thickness_comparison": control}
    for section, label in (("z4", "reference"), ("z6", "candidate")):
        point = "h80_" + section
        outputs = {"scattering_AP_pnc.csv": diagnosis.ROLE_PINS[point + "_scattering"][1],
                   "scattering_channels_AP_pnc.csv": diagnosis.ROLE_PINS[point + "_coefficients"][1]}
        control[label] = {"summary_sha256": diagnosis.ROLE_PINS[point + "_summary"][1],
                          "run_hash": point, "output_sha256": outputs}
        documents[point + "_summary"] = {"frequency_hz": 1e9, "incident_side": "left", "run_hash": point,
            "incident_mode_indices": list(diagnosis.FIXED_ROOTS[section]), "num_incident_channels": 5,
            "power_balance_passed": True, "output_sha256": outputs,
            "configuration": {"analysis": {"frequency_GHz": 1}, "geometry": {"thickness_nm": 200},
                "material": {"density_kg_m3": 5323, "temperature_policy": "fixed_property_control"}}}
        for group in ("incoming", "transmission", "reflection"):
            value = control["matched_channels"].setdefault(group, {})
            value[label + "_roots"] = list(diagnosis.FIXED_REVERSE[section] if group == "reflection" else diagnosis.FIXED_ROOTS[section])
            value[label + "_wavenumber_rad_m"] = ((-1 if group == "reflection" else 1) * np.arange(1, 6)).tolist()
    documents["z4_safe_receipt"] = {"status": "solved", "trace_dofs": 675, "inputs": {"frequency_hz": 1e9},
                                   "arrays": {"sha256": diagnosis.ROLE_PINS["z4_section"][1]}}
    documents["z6_safe_receipt"] = {"status": "safe_ready", "trace_dofs": 975, "frequency_hz": 1e9,
        "ports": {"sha256": diagnosis.ROLE_PINS["z6_section"][1]},
        "incident_port_roots": list(diagnosis.FIXED_ROOTS["z6"]), "backward_port_roots": list(diagnosis.FIXED_REVERSE["z6"])}
    documents["device_thickness_verification"] = {
        "h80_z4_z6_mesh_pair": {"comparison_summary_sha256": diagnosis.ROLE_PINS["thickness_comparison"][1],
            "mesh_pair_passed": False, "maximum_complex_change": control["maximum_complex_change"],
            "maximum_power_change": control["maximum_power_change"]},
        "safe_section": {"receipt_sha256": diagnosis.ROLE_PINS["z6_safe_receipt"][1],
                         "ports_file_sha256": diagnosis.ROLE_PINS["z6_section"][1]},
        "scattering_export": {"public_summary_sha256": diagnosis.ROLE_PINS["h80_z6_summary"][1]}}
    return documents


class ConversionFamiliesTests(TestCase):
    def test_five_modes_and_shuffled_tables_keep_signed_roles(self):
        modes = mode_fixture()
        summary, totals, coefficients, _, _ = response_fixture(modes)
        response = diagnosis.parse_response_tables(summary, totals[::-1], coefficients[::2] + coefficients[1::2], modes)
        self.assertEqual([row["root"] for row in response["rows"]], modes["roots"].tolist())
        self.assertEqual([row["reverse_root"] for row in response["rows"]], modes["reverse_roots"].tolist())
        self.assertEqual(len(response["coefficients"]), 50)
        np.testing.assert_allclose([row["converted_R"] for row in response["rows"]], [0, .0025, 0, 0, .0025])
        pair = diagnosis.select_conversion_pair(response)["selected_pair"]
        self.assertEqual(pair["forward_roots"], [4, 13])
        self.assertEqual([row["outgoing_root"] for row in pair["directions"]], [29, 8])
        self.assertAlmostEqual(pair["two_direction_power_sum"], .005)

    def test_missing_duplicate_and_wrong_signed_coefficients_reject(self):
        modes = mode_fixture()
        summary, totals, coefficients, _, _ = response_fixture(modes)
        for changed_totals, changed_coefficients in ((totals[:-1], coefficients), (totals, coefficients[:-1]),
                                                      (totals, coefficients[:-1] + coefficients[:1])):
            with self.assertRaises(ValueError):
                diagnosis.parse_response_tables(summary, changed_totals, changed_coefficients, modes)
        altered = copy.deepcopy(coefficients)
        altered[0]["outgoing_port_index"] = int(modes["roots"][0])
        with self.assertRaisesRegex(ValueError, "signed channel"):
            diagnosis.parse_response_tables(summary, totals, altered, modes)
        altered = copy.deepcopy(coefficients)
        altered[0]["incident_port_index"] = int(modes["roots"][1])
        with self.assertRaises(ValueError):
            diagnosis.parse_response_tables(summary, totals, altered, modes)

    def test_power_summary_and_total_corruption_reject(self):
        modes = mode_fixture()
        summary, totals, coefficients, _, _ = response_fixture(modes)
        altered = copy.deepcopy(coefficients)
        altered[0]["power_fraction"] += .001
        with self.assertRaisesRegex(ValueError, "complex amplitude"):
            diagnosis.parse_response_tables(summary, totals, altered, modes)
        altered_totals = copy.deepcopy(totals)
        altered_totals[0]["reflection"] += .001
        with self.assertRaisesRegex(ValueError, "power sums"):
            diagnosis.parse_response_tables(summary, altered_totals, coefficients, modes)
        for key, value in (("frequency_hz", 1e8), ("num_incident_channels", 4), ("incident_mode_indices", summary["incident_mode_indices"][::-1])):
            with self.assertRaises(ValueError):
                diagnosis.parse_response_tables({**summary, key: value}, totals, coefficients, modes)

    def test_reverse_wavenumber_pair_is_unique_and_not_root_adjacency(self):
        modes = mode_fixture()
        result = diagnosis.unique_reverse_partners(modes["roots"], modes["reverse_roots"][::-1], modes["waves"])
        np.testing.assert_array_equal(result, modes["reverse_roots"])
        waves = modes["waves"].copy()
        waves[modes["reverse_roots"][1]] = waves[modes["reverse_roots"][0]]
        with self.assertRaisesRegex(ValueError, "unique"):
            diagnosis.unique_reverse_partners(modes["roots"], modes["reverse_roots"], waves)

    def test_section_loader_accepts_all_five_modes_and_native_aliases(self):
        modes = mode_fixture()
        roots, reverse = modes["roots"], modes["reverse_roots"]
        rest = [i for i in range(30) if i not in roots and i not in reverse]
        positive = np.r_[roots, rest[:10]]
        negative = np.r_[reverse, rest[10:]]
        propagating, powers = np.zeros(30, dtype=bool), np.zeros(30)
        propagating[np.r_[roots, reverse]] = True
        powers[roots], powers[reverse] = 1, -1
        fields = np.ones((15, 30), dtype=complex)
        common = {"positive_indices": positive, "negative_indices": negative, "propagating": propagating,
                  "relative_residuals": np.zeros(30), "displacements": fields}
        with TemporaryDirectory() as directory:
            for name in ("z4", "z6"):
                path = Path(directory) / (name + ".npz")
                aliases = {"k_rad_m": modes["waves"], "signed_power_w": powers} if name == "z4" else {
                    "wavenumbers_rad_m": modes["waves"], "powers_w": powers, "frequency_hz": 1e9}
                np.savez(path, **common, **aliases)
                result = diagnosis.read_modes(path, name, roots, reverse, 15)
                self.assertEqual(result["displacements"].shape, (15, 5))
                np.testing.assert_array_equal(result["reverse_roots"], reverse)
                with self.assertRaises(ValueError):
                    diagnosis.read_modes(path, name, roots[:-1], reverse, 15)

    def test_two_way_conversion_and_complex_gauge_changes(self):
        modes = mode_fixture()
        summary, totals, coefficients, reflection, transmission = response_fixture(modes)
        reference = diagnosis.parse_response_tables(summary, totals, coefficients, modes)
        incoming_phase = np.array([.2, -.4, 1.1, .7, -.8])
        outgoing_phases = {"reflection": np.array([-.6, .1, .9, -1.2, .3]), "transmission": incoming_phase}
        candidate_coefficients = copy.deepcopy(coefficients)
        roots, reverse = modes["roots"].tolist(), modes["reverse_roots"].tolist()
        for row in candidate_coefficients:
            i = roots.index(row["incident_port_index"])
            j = (reverse if row["kind"] == "reflection" else roots).index(row["outgoing_port_index"])
            phase = incoming_phase[i] - outgoing_phases[row["kind"]][j]
            value = complex(row["amplitude_real"], row["amplitude_imag"]) * np.exp(1j * phase)
            row["amplitude_real"], row["amplitude_imag"] = value.real, value.imag
        candidate = diagnosis.parse_response_tables(summary, totals, candidate_coefficients, modes)
        control = {"matched_channels": {}, "blocks": {}}
        for group, group_roots, phase in (("incoming", roots, incoming_phase), ("reflection", reverse, outgoing_phases["reflection"]),
                                           ("transmission", roots, incoming_phase)):
            control["matched_channels"][group] = {"reference_roots": group_roots, "candidate_roots": group_roots,
                                                  "candidate_gauge_phase_rad": phase.tolist()}
        for kind, block in (("reflection", reflection), ("transmission", transmission)):
            packed = np.stack((block.real, block.imag), axis=-1).tolist()
            control["blocks"][kind] = {"reference": packed, "candidate_aligned": packed}
        changed = diagnosis.coefficient_changes({"h80_z4": reference, "h80_z6": candidate}, control)
        self.assertEqual(len(changed), 50)
        self.assertLess(max(row["absolute_complex_change"] for row in changed), 1e-15)
        nonzero = [row for row in changed if abs(complex(*row["z4_amplitude"])) > 0]
        self.assertGreater(max(abs(complex(*row["z6_raw_amplitude"]) - complex(*row["z4_amplitude"])) for row in nonzero), .1)
        changed_control = copy.deepcopy(control)
        changed_control["matched_channels"]["reflection"]["candidate_gauge_phase_rad"][1] += .1
        with self.assertRaisesRegex(ValueError, "gauge-aligned"):
            diagnosis.coefficient_changes({"h80_z4": reference, "h80_z6": candidate}, changed_control)

    def test_metadata_rejects_swapped_roles_and_erased_failed_control(self):
        documents = metadata_fixture()
        diagnosis.validate_metadata(documents)
        swapped = copy.deepcopy(documents)
        swapped["h80_z4_summary"], swapped["h80_z6_summary"] = swapped["h80_z6_summary"], swapped["h80_z4_summary"]
        with self.assertRaises(ValueError):
            diagnosis.validate_metadata(swapped)
        altered = copy.deepcopy(documents)
        altered["thickness_comparison"]["complex_passed"] = True
        with self.assertRaises(ValueError):
            diagnosis.validate_metadata(altered)
        altered = copy.deepcopy(documents)
        altered["h80_z4_summary"]["output_sha256"]["scattering_channels_AP_pnc.csv"] = diagnosis.ROLE_PINS["h80_z6_coefficients"][1]
        with self.assertRaises(ValueError):
            diagnosis.validate_metadata(altered)

    def test_input_roles_and_fresh_output_preserve_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text("a", encoding="utf-8")
            (root / "b.json").write_text("b", encoding="utf-8")
            pins = {"a": ("a.json", hashlib.sha256(b"a").hexdigest()), "b": ("b.json", hashlib.sha256(b"b").hexdigest())}
            with patch.object(diagnosis, "ROLE_PINS", pins):
                diagnosis.bind_input(root, "a", [])
                (root / "a.json").write_text("b", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "role a"):
                    diagnosis.bind_input(root, "a", [])
            row = {"root": 1, "reverse_root": 2, "wavenumber_rad_m": 3., "signed_power_w": 1., "R": .1, "T": .9,
                   "converted_R": .01, "converted_T": .02, "component_fractions_xyz": [1., 0., 0.],
                   "template_scores_tx_ty_tz_rx": [1., 0., 0., 0.], "template_projection_fraction": 1.,
                   "template_remainder_fraction": 0., "descriptive_largest_template": "Tx"}
            document = {"responses": {"h80_" + name: {"section": name, "rows": [dict(row, root=i) for i in range(5)]}
                                       for name in ("z4", "z6")}}
            output = root / "output"
            path = diagnosis.publish(document, output)
            before = path.read_bytes()
            with (output / diagnosis.CSV_NAME).open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 10)
            saved = json.loads(before)
            self.assertEqual(saved["output_sha256"][diagnosis.CSV_NAME], diagnosis.file_sha(output / diagnosis.CSV_NAME))
            with self.assertRaises(ValueError):
                diagnosis.publish(document, output)
            self.assertEqual(path.read_bytes(), before)
