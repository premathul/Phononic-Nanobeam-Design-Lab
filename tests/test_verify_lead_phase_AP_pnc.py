"""Portable independent coefficient/null-audit checks. Athul Prem | _AP_pnc."""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools import verify_lead_phase_AP_pnc as verify


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixture(root, scale=1.5, zero=False):
    """Construct raw response files, independent of producer/comparator helpers."""
    root = Path(root).resolve()
    source = root / "producer_AP_pnc.py"
    source.write_text('"""Synthetic executed producer identity."""\n', encoding="utf-8")
    basis = {"status": "identical_saved_basis", "incoming_roots": list(verify.INCOMING),
             "reflected_roots": list(verify.REFLECTED), "transmitted_roots": list(verify.INCOMING),
             "safe_modes": {"sha256": verify.SAFE_MODES_SHA},
             "safe_section": {"sha256": verify.SAFE_SECTION_SHA},
             "candidate_gauge_complex": [[1., 0.]] * 5, "candidate_gauge_phase_rad": [0.] * 5,
             "field_matching_recomputed": False}
    comparison = {"frequency_hz": verify.FREQUENCY, "reference_resolution": "h100_z4",
                  "candidate_resolution": "h80_z4", "complex_change_limit": .01,
                  "maximum_power_change": .0048, "maximum_complex_change": .043,
                  "power_change_limit": .01, "power_passed": True, "complex_passed": False,
                  "mesh_pair_passed": False, "pair_specific_targets": {"finite": "retained synthetic evidence"},
                  "other_prior_failed_spatial_comparisons_remain_unresolved": True,
                  "shared_basis": basis, "inputs": {}}
    values = {}
    for order, incident in enumerate(verify.INCOMING, 1):
        raw = {(kind, incident, outgoing): complex(.03 + .01 * order, .001 * index)
               for kind, roots in (("reflection", verify.REFLECTED), ("transmission", verify.INCOMING))
               for index, outgoing in enumerate(roots)}
        if zero and verify.PAIRS[0] in raw:
            raw[verify.PAIRS[0]] = 0j
        old_norm = math.sqrt(sum(abs(value)**2 for value in raw.values()))
        new_norm = math.sqrt(sum(abs(value * (scale if key[0] == "reflection" else 1.))**2
                                 for key, value in raw.items()))
        for key, value in raw.items():
            values[key] = value / old_norm, value * (scale if key[0] == "reflection" else 1.) / new_norm
    for label, dofs, position in (("h100", 307611, 0), ("h80", 426627, 1)):
        directory = root / label
        directory.mkdir()
        table = directory / "scattering_channels_AP_pnc.csv"
        fields = ["kind", "incident_mode_order", "incident_port_index", "outgoing_port_index",
                  "amplitude_real", "amplitude_imag", "power_fraction"]
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for kind, roots in (("reflection", verify.REFLECTED), ("transmission", verify.INCOMING)):
                for order, incident in enumerate(verify.INCOMING, 1):
                    for index, outgoing in enumerate(roots):
                        value = values[kind, incident, outgoing][position]
                        writer.writerow(dict(kind=kind, incident_mode_order=order, incident_port_index=incident,
                                             outgoing_port_index=outgoing, amplitude_real=value.real,
                                             amplitude_imag=value.imag, power_fraction=value.real**2 + value.imag**2))
        summary = {"frequency_hz": verify.FREQUENCY, "incident_side": "left",
                   "incident_mode_indices": list(verify.INCOMING), "num_incident_channels": 5,
                   "port_dofs": 675, "ndof": dofs, "end_face_dofs": {"xleft": 675, "xright": 675},
                   "scattering_power_normalization_available": True, "power_balance_passed": True,
                   "output_sha256": {table.name: verify.sha256(table)}}
        summary_path = directory / "summary_AP_pnc.json"
        dump(summary_path, summary)
        for role, path in ((f"{label}_8875_summary", summary_path), (f"{label}_8875_coefficients", table)):
            comparison["inputs"][role] = verify.artifact(path, root)
    for role, status in (("accepted_safe", "safe_ready"), ("accepted_prebody", "independently_verified_safe"),
                         ("accepted_point", "point_verified_within_saved_discretization"),
                         ("point_replay", "point_verified_within_saved_discretization")):
        receipt = {"status": status, "frequency_hz": verify.FREQUENCY}
        if role in ("accepted_point", "point_replay"):
            label = "h80" if role == "accepted_point" else "h100"
            receipt.update(public_summary_sha256=comparison["inputs"][f"{label}_8875_summary"]["sha256"],
                           relative_residual_limit=1e-9, backward_residual_limit=1e-9)
        path = root / f"{role}_AP_pnc.json"
        dump(path, receipt)
        comparison["inputs"][role] = verify.artifact(path, root)
    basis["safe_receipt"] = comparison["inputs"]["accepted_safe"]
    path = root / "comparison_AP_pnc.json"
    dump(path, comparison)
    comparison_sha = verify.sha256(path)
    # Build expected publication values explicitly from the defining synthetic values.
    entries = []
    for kind, roots in (("reflection", verify.REFLECTED), ("transmission", verify.INCOMING)):
        for order, incident in enumerate(verify.INCOMING, 1):
            for index, outgoing in enumerate(roots):
                a, b = values[kind, incident, outgoing]
                entries.append({"kind": kind, "incident_root": incident, "outgoing_root": outgoing,
                                "reference_raw_amplitude": [a.real, a.imag],
                                "candidate_raw_amplitude": [b.real, b.imag],
                                "reference_power": a.real**2 + a.imag**2,
                                "candidate_power": b.real**2 + b.imag**2,
                                "reference_magnitude": abs(a), "candidate_magnitude": abs(b),
                                "magnitude_lower_bound": abs(abs(b) - abs(a)),
                                "raw_complex_delta": [(b-a).real, (b-a).imag],
                                "absolute_raw_complex_change": abs(b-a),
                                "diagnostic_phase_change_rad": None if a == 0 else 0.,
                                "selected_directed_pair": (kind, incident, outgoing) in verify.PAIRS})
    maximum = max(row["magnitude_lower_bound"] for row in entries)
    publication = {"schema_version": 1, "status": "finite_phase_only_audit_completed",
                   "producer": verify.artifact(source, root), "accepted_comparison": verify.artifact(path, root),
                   "shared_basis": deepcopy(basis), "entries": entries,
                   "original_response_sources": {role: comparison["inputs"][role] for role in
                        ("h100_8875_summary", "h100_8875_coefficients", "h80_8875_summary", "h80_8875_coefficients")},
                   "preserved_original_acceptance": {name: comparison[name] for name in
                        ("maximum_power_change", "maximum_complex_change", "power_change_limit", "complex_change_limit",
                         "power_passed", "complex_passed", "mesh_pair_passed", "pair_specific_targets",
                         "other_prior_failed_spatial_comparisons_remain_unresolved")},
                   "scope": {"frequency_hz": verify.FREQUENCY, "reference_resolution": "h100_z4",
                             "candidate_resolution": "h80_z4", "incident_side": "left", "block_shape": [10, 5],
                             "coefficient_count": 50, "reference_planes_m": [0., 1.665e-5],
                             "native_incident_gauge_preserved": True,
                             **dict.fromkeys(("new_solve_performed", "new_assembly_performed", "arrays_read", "phase_fitting_performed",
                                              "phase_adjustment_applied", "deembedding_performed", "uniform_phase_subtraction_performed",
                                              "spatial_convergence_established", "numerical_floor_established"), False)},
                   "method": {"tested_model": "Diagonal, reflectionless, unmixed, unit-modulus lead propagation."},
                   "selected_pair": [deepcopy(row) for row in entries if row["selected_directed_pair"]],
                   "aggregate": {"maximum_magnitude_lower_bound": maximum,
                                 "frobenius_magnitude_lower_bound": math.sqrt(sum(row["magnitude_lower_bound"]**2 for row in entries)),
                                 "existing_complex_change_limit": .01, "bound_strictly_exceeds_existing_limit": maximum > .01,
                                 "frobenius_acceptance_gate_applied": False,
                                 "maximum_bound_entries": [{name: row[name] for name in ("kind", "incident_root", "outgoing_root")}
                                                           for row in entries if row["magnitude_lower_bound"] == maximum],
                                 "verdict": "phase-only model excluded at existing tolerance" if maximum > .01 else "not excluded"}}
    diagnostic = root / "lead_phase_AP_pnc.json"
    write_csv(root, publication)
    dump(diagnostic, publication)
    return diagnostic, publication, comparison, comparison_sha


def write_csv(root, publication):
    table = root / "lead_phase_AP_pnc.csv"
    columns = ("kind", "incident_root", "outgoing_root", "reference_real", "reference_imag", "candidate_real",
               "candidate_imag", "reference_power", "candidate_power", "reference_magnitude", "candidate_magnitude",
               "magnitude_lower_bound", "raw_complex_delta_real", "raw_complex_delta_imag", "absolute_raw_complex_change",
               "diagnostic_phase_change_rad", "selected_directed_pair")
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for entry in publication["entries"]:
            row = {name: entry[name] for name in columns if name in entry}
            for name, prefix in (("reference_raw_amplitude", "reference"), ("candidate_raw_amplitude", "candidate"),
                                 ("raw_complex_delta", "raw_complex_delta")):
                row[prefix + "_real"], row[prefix + "_imag"] = entry[name]
            writer.writerow(row)
    publication["publication"] = {"json": "lead_phase_AP_pnc.json", "csv": table.name,
                                   "html": "report_AP_pnc.html", "csv_sha256": verify.sha256(table)}


class IndependentLeadPhaseTests(unittest.TestCase):
    def run_fixture(self, root, diagnostic, publication, accepted_sha):
        dump(diagnostic, publication)
        return verify.verify_publication(root, diagnostic, publication["producer"]["sha256"], accepted_sha)

    def test_full_raw_reconstruction_and_two_selected_pairs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            diagnostic, publication, _, accepted_sha = fixture(root)
            result = self.run_fixture(root, diagnostic, publication, accepted_sha)
            self.assertEqual(result["verified_coefficient_rows"], 50)
            self.assertEqual(len(result["selected_directed_pairs"]), 2)
            self.assertGreater(result["maximum_magnitude_lower_bound"], .01)
            self.assertEqual(result["verdict"], "phase-only model excluded at existing tolerance")

    def test_zero_coefficients_keep_undefined_phase_and_no_relative_floor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            diagnostic, publication, _, accepted_sha = fixture(root, scale=1., zero=True)
            result = self.run_fixture(root, diagnostic, publication, accepted_sha)
            self.assertEqual(result["verdict"], "not excluded")
            self.assertEqual(result["maximum_magnitude_lower_bound"], 0.)
            selected = next(row for row in result["selected_directed_pairs"] if row["incident_root"] == 1302)
            self.assertIsNone(selected["diagnostic_phase_change_rad"])

    def test_false_verdict_or_loosened_gate_rejected(self):
        for update in ({"verdict": "not excluded"}, {"existing_complex_change_limit": .1}):
            with self.subTest(update=update), TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                diagnostic, publication, _, accepted_sha = fixture(root)
                publication["aggregate"].update(update)
                with self.assertRaises(ValueError):
                    self.run_fixture(root, diagnostic, publication, accepted_sha)

    def test_tampered_bound_and_aggregate_rejected(self):
        for field in ("row", "maximum", "frobenius"):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                diagnostic, publication, _, accepted_sha = fixture(root)
                if field == "row":
                    publication["entries"][0]["magnitude_lower_bound"] = 0.
                else:
                    name = "maximum_magnitude_lower_bound" if field == "maximum" else "frobenius_magnitude_lower_bound"
                    publication["aggregate"][name] = 0.
                with self.assertRaises(ValueError):
                    self.run_fixture(root, diagnostic, publication, accepted_sha)

    def test_basis_mismatch_and_duplicate_publication_rows_rejected(self):
        for change in ("basis", "duplicate", "missing", "phase"):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                diagnostic, publication, _, accepted_sha = fixture(root)
                if change == "basis":
                    publication["shared_basis"]["candidate_gauge_complex"][0] = [0., 1.]
                elif change == "duplicate":
                    publication["entries"][0] = publication["entries"][1]
                elif change == "missing":
                    publication["entries"].pop()
                else:
                    publication["entries"][0]["diagnostic_phase_change_rad"] = .1
                with self.assertRaises(ValueError):
                    self.run_fixture(root, diagnostic, publication, accepted_sha)

    def test_external_source_hash_and_raw_tamper_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            diagnostic, publication, comparison, accepted_sha = fixture(root)
            with self.assertRaises(ValueError):
                verify.verify_publication(root, diagnostic, "0" * 64, accepted_sha)
            table = root / comparison["inputs"]["h80_8875_coefficients"]["path"]
            table.write_text(table.read_text() + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.run_fixture(root, diagnostic, publication, accepted_sha)

    def test_raw_duplicate_unknown_order_and_power_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, _, comparison, _ = fixture(root)
            summary = verify.read_json(root / comparison["inputs"]["h100_8875_summary"]["path"])
            table = root / comparison["inputs"]["h100_8875_coefficients"]["path"]
            with table.open(newline="", encoding="utf-8") as stream:
                original = list(csv.DictReader(stream))
            for change in ("duplicate", "root", "order", "power", "nan"):
                with self.subTest(change=change):
                    rows = deepcopy(original)
                    if change == "duplicate":
                        rows[0] = rows[1]
                    elif change == "root":
                        rows[0]["outgoing_port_index"] = "999"
                    elif change == "order":
                        rows[0]["incident_mode_order"] = "2"
                    elif change == "power":
                        rows[0]["power_fraction"] = "1"
                    else:
                        rows[0]["amplitude_real"] = "nan"
                    with self.assertRaises(ValueError):
                        verify.parse_response(summary, rows, 307611)

    def test_publication_csv_tampering_rejected_even_with_updated_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            diagnostic, publication, _, accepted_sha = fixture(root)
            table = root / "lead_phase_AP_pnc.csv"
            text = table.read_text(encoding="utf-8")
            table.write_text(text.replace("reflection,1302,1303,", "reflection,1302,9999,", 1), encoding="utf-8")
            publication["publication"]["csv_sha256"] = verify.sha256(table)
            with self.assertRaises(ValueError):
                self.run_fixture(root, diagnostic, publication, accepted_sha)

    def test_selected_pair_and_preserved_acceptance_tampering_rejected(self):
        for change in ("selected", "acceptance", "model"):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                diagnostic, publication, _, accepted_sha = fixture(root)
                if change == "selected":
                    publication["selected_pair"][0]["magnitude_lower_bound"] = 0.
                elif change == "acceptance":
                    publication["preserved_original_acceptance"]["complex_passed"] = True
                else:
                    publication["method"]["tested_model"] = "All lead error is excluded."
                with self.assertRaises(ValueError):
                    self.run_fixture(root, diagnostic, publication, accepted_sha)

    def test_wrong_accepted_hash_and_outside_workspace_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            diagnostic, publication, _, accepted_sha = fixture(root)
            with self.assertRaises(ValueError):
                verify.verify_publication(root, diagnostic, publication["producer"]["sha256"], "0" * 64)
            with self.assertRaises(ValueError):
                verify.bounded(diagnostic, root / "h100")


if __name__ == "__main__":
    unittest.main()
