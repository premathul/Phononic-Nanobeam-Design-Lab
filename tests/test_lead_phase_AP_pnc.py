"""Portable complete-block phase-bound checks. Athul Prem | _AP_pnc."""
from __future__ import annotations

import cmath
from copy import deepcopy
import csv
from io import StringIO
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import diagnose_lead_phase_AP_pnc as audit


def fixture():
    rows = []
    for i, incoming in enumerate(audit.ROOTS):
        magnitudes = [.05+.003*(j+1) for j in range(5)] + [.4+.01*(j+1) for j in range(5)]
        scale = math.sqrt(math.fsum(m*m for m in magnitudes))
        for block, roots in enumerate((audit.REVERSE, audit.ROOTS)):
            for j, outgoing in enumerate(roots):
                a = magnitudes[block*5+j]/scale*cmath.exp(1j*(.17*i-.11*j))
                b = a*cmath.exp(1j*(.07*i+.13*j+.3*block+.1))
                row = {"kind": "reflection" if block == 0 else "transmission", "incident_root": incoming,
                       "outgoing_root": outgoing, "reference_raw_amplitude": [a.real, a.imag]}
                set_candidate(row, b)
                rows.append(row)
    return {"schema_version": 1, "phase": "device_inplane_comparison", "status": "compared_finite_inplane_increment",
            "frequency_hz": audit.FREQUENCY, "reference_resolution": "h100_z4", "candidate_resolution": "h80_z4",
            "input_power_checks_passed": True, "power_passed": True, "complex_passed": False, "mesh_pair_passed": False,
            "power_change_limit": .01, "complex_change_limit": .01, "maximum_power_change": 0.,
            "maximum_complex_change": max(r["raw_complex_change"]["absolute_complex_change"] for r in rows),
            "phase_adjustment_applied": False, "field_matching_recomputed": False, "spatial_convergence_established": False,
            "frequency_convergence_established": False, "numerical_floor_established": False,
            "other_prior_failed_spatial_comparisons_remain_unresolved": True, "pair_specific_targets": {},
            "shared_basis": {"status": "identical_saved_basis", "incoming_roots": list(audit.ROOTS), "transmitted_roots": list(audit.ROOTS),
                             "reflected_roots": list(audit.REVERSE), "candidate_gauge_complex": [[1., 0.]]*5,
                             "candidate_gauge_phase_rad": [0.]*5, "field_matching_recomputed": False,
                             "safe_modes": {"sha256": audit.MODES_SHA}, "safe_section": {"sha256": audit.SECTION_SHA},
                             "safe_receipt": {"sha256": audit.SAFE_SHA}},
            "inputs": {key: {"path": key+".json", "sha256": "a"*64, "bytes": 1} for key in audit.SOURCE_ROLES}, "coefficients": rows}


def set_candidate(row, b):
    a = complex(*row["reference_raw_amplitude"])
    delta = b-a
    row.update(candidate_raw_amplitude=[b.real, b.imag], candidate_gauge_aligned_amplitude=[b.real, b.imag],
               reference_power=abs(a)**2, candidate_power=abs(b)**2)
    row["raw_complex_change"] = {"complex_delta": [delta.real, delta.imag], "absolute_complex_change": abs(delta),
                                 "phase_change_rad": cmath.phase(b/a) if a != 0 and b != 0 else None}
    row["gauge_aligned_complex_change"] = deepcopy(row["raw_complex_change"])


def refresh_max(value):
    value["maximum_complex_change"] = max(r["raw_complex_change"]["absolute_complex_change"] for r in value["coefficients"])


class AlgebraTests(unittest.TestCase):
    def test_all_fifty_phase_only_rows_are_not_excluded_despite_complex_failure(self):
        result = audit.audit(fixture())
        self.assertEqual(len(result["entries"]), 50)
        self.assertEqual(result["aggregate"]["verdict"], "not excluded")
        self.assertLess(result["aggregate"]["maximum_magnitude_lower_bound"], 2e-16)
        self.assertEqual(len(result["selected_pair"]), 2)
        self.assertGreater(max(r["absolute_raw_complex_change"] for r in result["entries"]), .01)

    def test_magnitude_change_excludes_model_and_bound_is_valid_for_every_phase(self):
        value = fixture()
        first, second = value["coefficients"][0:2]
        p = first["candidate_power"]+second["candidate_power"]
        set_candidate(first, math.sqrt(p)*cmath.exp(.9j))
        set_candidate(second, 0j)
        refresh_max(value)
        result = audit.audit(value)
        self.assertEqual(result["aggregate"]["verdict"], "phase-only model excluded at existing tolerance")
        self.assertGreater(result["aggregate"]["maximum_magnitude_lower_bound"], .01)
        for row in result["entries"]:
            a, b = complex(*row["reference_raw_amplitude"]), complex(*row["candidate_raw_amplitude"])
            for theta in (-math.pi, -1.2, 0., .7, math.pi):
                self.assertGreaterEqual(abs(b-cmath.exp(1j*theta)*a)+2e-16, row["magnitude_lower_bound"])
        self.assertIsNone(result["entries"][1]["diagnostic_phase_change_rad"])
        expected = math.sqrt(math.fsum(r["magnitude_lower_bound"]**2 for r in result["entries"]))
        self.assertAlmostEqual(result["aggregate"]["frobenius_magnitude_lower_bound"], expected, places=15)

    def test_zero_reference_and_zero_zero_have_undefined_phase(self):
        value = fixture()
        row = value["coefficients"][0]
        row["reference_raw_amplitude"] = [0., 0.]
        set_candidate(row, 0j)
        refresh_max(value)
        result = audit.audit(value)
        self.assertIsNone(result["entries"][0]["diagnostic_phase_change_rad"])
        self.assertEqual(result["entries"][0]["magnitude_lower_bound"], 0)
        set_candidate(row, .03j)
        refresh_max(value)
        self.assertIsNone(audit.audit(value)["entries"][0]["diagnostic_phase_change_rad"])

    def test_bound_equal_to_existing_limit_does_not_exclude_model(self):
        value = fixture()
        row = value["coefficients"][0]
        row["reference_raw_amplitude"] = [0., 0.]
        set_candidate(row, .01j)
        refresh_max(value)
        result = audit.audit(value)
        self.assertEqual(result["aggregate"]["maximum_magnitude_lower_bound"], .01)
        self.assertFalse(result["aggregate"]["bound_strictly_exceeds_existing_limit"])
        self.assertEqual(result["aggregate"]["verdict"], "not excluded")

    def test_bound_invariant_to_independent_phases_and_row_permutation(self):
        first = fixture()
        row = first["coefficients"][0]
        set_candidate(row, complex(*row["candidate_raw_amplitude"])*1.1)
        refresh_max(first)
        second = deepcopy(first)
        for n, row in enumerate(second["coefficients"]):
            a, b = complex(*row["reference_raw_amplitude"]), complex(*row["candidate_raw_amplitude"])
            a *= cmath.exp(1j*(n*.21+.4))
            b *= cmath.exp(1j*(n*.31-.2))
            row["reference_raw_amplitude"] = [a.real, a.imag]
            set_candidate(row, b)
        second["coefficients"].reverse()
        refresh_max(second)
        a, b = audit.audit(first), audit.audit(second)
        for key in ("maximum_magnitude_lower_bound", "frobenius_magnitude_lower_bound"):
            self.assertAlmostEqual(a["aggregate"][key], b["aggregate"][key], places=14)
        self.assertEqual([(r["kind"], r["incident_root"], r["outgoing_root"]) for r in a["entries"]],
                         [(r["kind"], r["incident_root"], r["outgoing_root"]) for r in b["entries"]])

    def test_corrupt_rows_powers_phase_and_nonfinite_values_are_rejected(self):
        corruptions = (lambda v: v["coefficients"].pop(),
                       lambda v: v["coefficients"].__setitem__(1, deepcopy(v["coefficients"][0])),
                       lambda v: v["coefficients"][0].__setitem__("incident_root", -1),
                       lambda v: v["coefficients"][0].__setitem__("reference_power", .1),
                       lambda v: v["coefficients"][0].__setitem__("candidate_power", -1),
                       lambda v: v["coefficients"][0]["candidate_raw_amplitude"].__setitem__(0, float("nan")),
                       lambda v: v["coefficients"][0]["raw_complex_change"].__setitem__("phase_change_rad", .9),
                       lambda v: v["coefficients"][0]["candidate_gauge_aligned_amplitude"].__setitem__(0, .9))
        for mutate in corruptions:
            value = fixture()
            mutate(value)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                audit.audit(value)

    def test_changed_signed_basis_gauges_and_original_gates_are_rejected(self):
        corruptions = (lambda v: v["shared_basis"]["incoming_roots"].reverse(),
                       lambda v: v["shared_basis"]["candidate_gauge_complex"].__setitem__(0, [0., 1.]),
                       lambda v: v["shared_basis"]["candidate_gauge_phase_rad"].__setitem__(0, .1),
                       lambda v: v["shared_basis"]["safe_modes"].__setitem__("sha256", "b"*64),
                       lambda v: v.__setitem__("complex_change_limit", .02),
                       lambda v: v.__setitem__("complex_passed", True),
                       lambda v: v.__setitem__("phase_adjustment_applied", True))
        for mutate in corruptions:
            value = fixture()
            mutate(value)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                audit.audit(value)


class PublicationTests(unittest.TestCase):
    def test_fresh_publication_roundtrip_sources_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root/"comparison.json"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            own_source = root/"producer.py"
            own_source.write_text("synthetic producer identity\n", encoding="utf-8")
            output = root/"diagnostic"
            with patch.object(audit, "COMPARISON_SHA", audit.sha(source)), patch.object(audit, "__file__", str(own_source)):
                result = audit.run(root, source, output)
                with self.assertRaisesRegex(ValueError, "fresh"):
                    audit.run(root, source, output)
            self.assertEqual(set(p.name for p in output.iterdir()), {audit.JSON_NAME, audit.CSV_NAME, audit.HTML_NAME})
            saved = json.loads((output/audit.JSON_NAME).read_text(encoding="utf-8"))
            self.assertEqual(saved["shared_basis"], fixture()["shared_basis"])
            self.assertEqual(saved["producer"]["sha256"], audit.sha(own_source))
            self.assertEqual(saved["publication"]["csv_sha256"], audit.sha(output/audit.CSV_NAME))
            rows = list(csv.DictReader(StringIO((output/audit.CSV_NAME).read_text(encoding="utf-8"))))
            self.assertEqual(len(rows), 50)
            self.assertEqual(set(rows[0]), set(audit.CSV_COLUMNS))
            self.assertEqual(float(rows[0]["magnitude_lower_bound"]), result["entries"][0]["magnitude_lower_bound"])
            self.assertIn("not excluded", (output/audit.HTML_NAME).read_text(encoding="utf-8"))
            self.assertEqual(saved["accepted_comparison"]["path"], "comparison.json")

    def test_changed_hash_is_rejected_without_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root/"comparison.json"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                audit.run(root, source, root/"out")
            self.assertFalse((root/"out").exists())


if __name__ == "__main__":
    unittest.main()
