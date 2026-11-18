"""Synthetic saved-response contract checks. Athul Prem | _AP_pnc."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from tools import compare_frequency_inplane_AP_pnc as comparison


def synthetic_response(power=.001):
    coefficients, rows = [], []
    for incoming, reverse in zip(comparison.ROOTS, comparison.REVERSE, strict=True):
        for kind, outgoing in (("reflection", comparison.REVERSE), ("transmission", comparison.ROOTS)):
            for root in outgoing:
                value = power if kind == "reflection" and (incoming, root) in comparison.PAIR else 0.
                if kind == "transmission" and root == incoming:
                    value = 1 - (power if incoming in (comparison.PAIR[0][0], comparison.PAIR[1][0]) else 0.)
                coefficients.append({"kind": kind, "incident_root": incoming, "outgoing_root": root,
                                     "amplitude": [value**.5, 0.], "power": value})
        reflected = power if incoming in (comparison.PAIR[0][0], comparison.PAIR[1][0]) else 0.
        rows.append({"root": incoming, "reverse_root": reverse, "R": reflected, "T": 1-reflected,
                     "converted_R": reflected, "converted_T": 0., "total_converted_power": reflected})
    return {"run_hash": "synthetic", "frequency_hz": comparison.FREQUENCY, "coefficients": coefficients, "rows": rows}


def make_fixture(root):
    """Create small receipts and all raw rows; never open a scientific array."""
    pins = {}

    def save(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return comparison.artifact(path, root)

    def pin(role, value):
        relative = f"pins/{role}.json"
        link = save(relative, value)
        pins[role] = relative, link["sha256"]
        return link

    sources = {}
    for role in ("response_reader_source", "accepted_point_source", "accepted_prebody_source", "accepted_relation_source"):
        path = root / "sources" / (role + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic producer identity.\n", encoding="utf-8")
        sources[role] = comparison.artifact(path, root)
        pins[role] = path.relative_to(root).as_posix(), sources[role]["sha256"]
    verifier = root / "tools/replay_lawrence_frequency_h100_AP_pnc.py"
    verifier.parent.mkdir(parents=True, exist_ok=True)
    verifier.write_text("# Synthetic new verifier.\n", encoding="utf-8")
    modes = {"path": "modes.npz", "bytes": 19, "sha256": comparison.SAFE_MODES_SHA}
    section = {"path": "section.npz", "bytes": 21, "sha256": comparison.SAFE_SECTION_SHA}
    receipt = {"path": "section.json", "bytes": 23, "sha256": comparison.SECTION_RECEIPT_SHA}
    opened = [{"root_id": root_id, "direction": direction, "k_imag_rad_m": 0., "k_real_rad_m": sign * (index+1.),
               "signed_power_w": float(sign)}
              for direction, sign, roots in (("positive", 1, comparison.ROOTS), ("negative", -1, comparison.REVERSE))
              for index, root_id in enumerate(roots)]
    safe = {"status": "safe_ready", "frequency_hz": comparison.FREQUENCY, "discrete_roots": 1350,
            "positive_open_channels": 5, "negative_open_channels": 5, "safe_modes": modes,
            "safe_section": section, "section_receipt": receipt, "reference_planes_m": comparison.PLANES,
            "degeneracy_status": "none_detected", "open_channels": opened}
    safe_link = pin("accepted_safe", safe)
    relation = {"status": "signed_section_relation_accepted", "root_permutation_square": {"status": "consistent"},
                "producer": sources["accepted_relation_source"], "edges": {"z4_frequency": {"status": "matched", "matched_channels": {
                    "incoming": {"reference_roots": comparison.ANCHOR_ROOTS, "candidate_roots": [1346, 1302, 1292, 1286, 1282]},
                    "reflection": {"reference_roots": comparison.ANCHOR_REVERSE, "candidate_roots": [1347, 1303, 1293, 1287, 1283]}}}}}
    relation_link = pin("accepted_relation", relation)
    prebody_link = pin("accepted_prebody", {"status": "independently_verified_safe", "relation_status": "matched",
        "safe_receipt": safe_link, "safe_modes": modes, "relation": relation_link,
        "implementation": {"verifier": sources["accepted_prebody_source"]}})

    def response(name, dofs, size, power, frequency=comparison.FREQUENCY):
        path = root / name / "summary_AP_pnc.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = synthetic_response(power)
        old = frequency == comparison.ANCHOR_FREQUENCY
        roots, reverse = (comparison.ANCHOR_ROOTS, comparison.ANCHOR_REVERSE) if old else (comparison.ROOTS, comparison.REVERSE)
        forward_map = dict(zip(comparison.ROOTS, roots, strict=True))
        backward_map = dict(zip(comparison.REVERSE, reverse, strict=True))
        if old:
            # Preserve pair identity through the saved frequency permutation.
            forward_map = {1346:1348, 1302:1311, 1292:1292, 1286:1284, 1282:1274}
            backward_map = {1347:1349, 1303:1312, 1293:1293, 1287:1285, 1283:1275}
        total_rows, coefficient_rows = [], []
        for index, incoming in enumerate(roots, 1):
            original = next(key for key, value in forward_map.items() if value == incoming)
            drive = next(row for row in data["rows"] if row["root"] == original)
            total_rows.append({"incident_mode_order": index, "frequency_GHz": frequency/1e9,
                               "reflection": drive["R"], "transmission": drive["T"]})
        for row in data["coefficients"]:
            incoming, outgoing = forward_map[row["incident_root"]], (backward_map if row["kind"] == "reflection" else forward_map)[row["outgoing_root"]]
            coefficient_rows.append({"kind": row["kind"], "incident_mode_order": roots.index(incoming)+1,
                "incident_port_index": incoming, "outgoing_port_index": outgoing,
                "amplitude_real": row["amplitude"][0], "amplitude_imag": 0., "power_fraction": row["power"]})
        output = {"port_modes_AP_pnc.csv": comparison.SHARED_PORT_TABLE_SHA}
        for filename, rows in (("scattering_AP_pnc.csv", total_rows), ("scattering_channels_AP_pnc.csv", coefficient_rows)):
            with (path.parent / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(reversed(rows))
            output[filename] = comparison.sha(path.parent / filename)
        provenance = {"reference_planes_m": comparison.PLANES, "phase_adjustment_applied": False,
            "safe_receipt": safe_link, "safe_modes": modes, "safe_section": section, "section_receipt": receipt,
            "open_channels": opened, "prepared_frequency_stage": {"path": "stage.json", "bytes": 5, "sha256": "a"*64},
            "solve_receipt": {"path": "solve.json", "bytes": 7, "sha256": "b"*64}}
        summary = {"status": "completed_unverified", "ndof": dofs, "port_dofs": 675, "hole_count": 26,
            "configuration": {"geometry": {"width_nm": 1000}, "material": {"name": "synthetic"},
                "mesh": {"mesh_size_nm": size, "thickness_layers": 4}}, "hole_pattern_provenance": {"source": "synthetic"},
            "checkpointed_scattering": provenance, "frequency_hz": frequency, "incident_side": "left",
            "incident_mode_indices": roots, "num_incident_channels": 5, "power_balance_passed": True,
            "run_hash": name, "output_sha256": output,
            "implementation": {"packages": {"synthetic": "1"}, "source_sha256": {
                f"src/phononic/{stem}_AP_pnc.py": "c"*64 for stem in (*comparison.response_reader.matching.CORE, "layered", "assembly_batched")}}}
        link = save(path.relative_to(root), summary)
        return path, summary, link

    h80_path, h80_summary, h80_link = response("h80", 426627, 80, .001)
    h100_path, h100_summary, h100_link = response("h100", 307611, 100, .0012)
    anchors = {name: response(name + "_anchor", dofs, size, .2, comparison.ANCHOR_FREQUENCY)
               for name, dofs, size in (("h100", 307611, 100), ("h80", 426627, 80))}
    for name, (path, _, link) in anchors.items():
        pins[name + "_anchor_summary"] = path.relative_to(root).as_posix(), link["sha256"]
    anchor_proof = {"status": "compared", "complex_passed": False, "mesh_pair_passed": False, "matched_channels": {}}
    for name, role in (("h100", "reference"), ("h80", "candidate")):
        _, summary, link = anchors[name]
        anchor_proof[role] = {"summary_sha256": link["sha256"], "run_hash": summary["run_hash"], "output_sha256": summary["output_sha256"]}
    for group, roots in (("incoming", comparison.ANCHOR_ROOTS), ("transmission", comparison.ANCHOR_ROOTS), ("reflection", comparison.ANCHOR_REVERSE)):
        anchor_proof["matched_channels"][group] = {"reference_roots": roots, "candidate_roots": roots,
            "mass_overlaps": [1.]*5, "candidate_gauge_phase_rad": [0.]*5,
            "reference_wavenumber_rad_m": [1.,2.,3.,4.,5.], "candidate_wavenumber_rad_m": [1.,2.,3.,4.,5.]}
    pin("accepted_anchor_relation", anchor_proof)

    def replay(summary_path, summary, dofs, producer):
        return {"status": "point_verified_within_saved_discretization", "frequency_hz": comparison.FREQUENCY,
            "problem_dofs": dofs, "port_dofs": 675, "reference_planes_m": comparison.PLANES,
            "public_summary": comparison.artifact(summary_path, root), "public_summary_sha256": comparison.sha(summary_path),
            "relative_target": 1e-10, "relative_residual_limit": 1e-9, "backward_residual_limit": 1e-9,
            "maximum_relative_residual": 1e-12, "maximum_backward_residual": 1e-15,
            "propagating_coefficients_checked": 50, "full_directional_coefficients_checked": 6750,
            **dict.fromkeys(("original_equations_passed", "physical_nodal_force_power_passed", "all_modal_coefficients_passed",
                            "full_p2_field_identity_passed", "coherent_left_incidence_passed"), True),
            "coherent_left_incidence": {"incident_roots": comparison.ROOTS, "transmitted_roots": comparison.ROOTS, "reflected_roots": comparison.REVERSE},
            "channels": [{"column": i, "incident_root": a, "reverse_root": b} for i,(a,b) in enumerate(zip(comparison.ROOTS,comparison.REVERSE,strict=True))],
            "phase_receipts": {"safe": safe_link, "stage": summary["checkpointed_scattering"]["prepared_frequency_stage"],
                               "solve": summary["checkpointed_scattering"]["solve_receipt"]},
            "accepted_shared_safe": safe_link, "prebody_replay": prebody_link, "relation": relation_link,
            "implementation": {"verifier": producer}}
    accepted = replay(h80_path, h80_summary, 426627, sources["accepted_point_source"])
    accepted_link = pin("accepted_point", accepted)
    point = replay(h100_path, h100_summary, 307611, comparison.artifact(verifier, root))
    point["accepted_reference_point"] = accepted_link
    point_link = save("point.json", point)
    pin("accepted_thickness_comparison", {"status": "compared_finite_thickness_increment", "complex_passed": False,
        "maximum_complex_change": .01753014, "pair_specific_targets": {"directed_increment_passed": False},
        "other_prior_failed_spatial_comparisons_remain_unresolved": True,
        "inputs": {"z4_8875_summary": h80_link}})
    return {"pins": pins, "point": point, "point_path": root/"point.json", "point_sha": point_link["sha256"],
            "verifier_sha": comparison.sha(verifier), "candidate": h100_path.parent, "summary": h100_summary}


class FrequencyInplaneTests(TestCase):
    def test_directed_targets_use_exact_positive_h80_denominators_without_averaging(self):
        value = comparison.pair_targets([.001, .0012], [.001, .001], [.2,.2], [.2,.2])
        self.assertFalse(value["directed_increment_passed"])
        self.assertEqual([row["passed"] for row in value["directed_increment_rows"]], [True, False])
        self.assertEqual(value["directed_increment_rows"][1]["exact_denominator"], .001)
        self.assertFalse(value["denominator_floor_applied"])
        tiny = comparison.pair_targets([2e-30, 1e-30], [1e-30, 1e-30], [1.,1.], [1.,1.])
        self.assertEqual(tiny["directed_increment_rows"][0]["relative_absolute_power_change"], 1.)

    def test_suppression_has_four_independent_tests(self):
        value = comparison.pair_targets([.02,.001], [.02,.001], [1.,1.], [10.,1.])
        self.assertTrue(value["directed_increment_passed"])
        self.assertFalse(value["suppression_retention_passed"])
        self.assertEqual(sum(not row["passed"] for row in value["suppression_retention_rows"]), 1)

    def test_invalid_denominators_and_nonfinite_values_fail(self):
        for value in (0., -1., float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                comparison.pair_targets([.001,.001], [value,.001], [1.,1.], [1.,1.])
        with self.assertRaises(ValueError):
            comparison.pair_targets([.001,.001], [.001,.001], [0.,1.], [1.,1.])

    def test_zero_coefficients_have_undefined_relative_change_and_phase(self):
        zero = comparison.complex_change(0j, 1j)
        self.assertIsNone(zero["relative_absolute_complex_change"])
        self.assertIsNone(zero["phase_change_rad"])
        self.assertEqual(zero["absolute_complex_change"], 1.)
        self.assertIsNone(comparison.complex_change(1j, 0j)["phase_change_rad"])
        self.assertIsNone(comparison.power_change(0.,1.)["relative_power_change"])

    def test_complete_coefficient_join_uses_signed_identity_and_exact_basis(self):
        first, second = synthetic_response(), synthetic_response(.0012)
        second["coefficients"].reverse()
        value = comparison.compare_responses(first, second)
        self.assertEqual(len(value["coefficients"]), 50)
        self.assertEqual(len(value["per_drive"]), 25)
        self.assertEqual(len(value["selected_pair"]), 2)
        self.assertNotIn("selected_pair_two_drive_power_sum", value)
        for row in value["coefficients"]:
            self.assertEqual(row["raw_complex_change"], row["gauge_aligned_complex_change"])
            self.assertEqual(row["candidate_raw_amplitude"], row["candidate_gauge_aligned_amplitude"])
        second["coefficients"][0]["outgoing_root"] = 999
        with self.assertRaisesRegex(ValueError, "identities"):
            comparison.compare_responses(first, second)

    def build(self, root, fixture):
        with patch.object(comparison, "PINS", fixture["pins"]), patch.object(comparison, "H80_POWERS", (.001,.001)), patch.object(comparison, "ANCHOR_POWERS", {"h100":(.2,.2),"h80":(.2,.2)}):
            return comparison.build_comparison(fixture["candidate"], fixture["point_path"], fixture["point_sha"], fixture["verifier_sha"], root)

    def test_synthetic_end_to_end_accepts_failed_target_as_completed_comparison(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            value = self.build(root, make_fixture(root))
            self.assertEqual(value["status"], "compared_finite_inplane_increment")
            self.assertFalse(value["pair_specific_targets"]["directed_increment_passed"])
            self.assertTrue(value["pair_specific_targets"]["suppression_retention_passed"])
            self.assertFalse(value["spatial_convergence_established"])
            self.assertEqual(value["shared_basis"]["status"], "identical_saved_basis")

    def test_externally_pinned_replay_and_source_are_mandatory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = make_fixture(root)
            for key in ("point_sha", "verifier_sha"):
                changed = deepcopy(fixture)
                changed[key] = "0"*64
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "externally accepted"):
                    self.build(root, changed)

    def test_rehashed_candidate_replay_cannot_swap_its_producer_or_accepted_reference(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = make_fixture(root)
            for role in ("implementation", "accepted_reference_point"):
                changed = deepcopy(fixture["point"])
                if role == "implementation":
                    changed[role]["verifier"]["sha256"] = "0"*64
                else:
                    changed[role]["sha256"] = "0"*64
                fixture["point_path"].write_text(json.dumps(changed), encoding="utf-8")
                fixture["point_sha"] = comparison.sha(fixture["point_path"])
                with self.subTest(role=role), self.assertRaises(ValueError):
                    self.build(root, fixture)

    def test_shared_basis_requires_fields_and_original_planes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = make_fixture(root)
            candidate = fixture["summary"]
            safe = json.loads((root / fixture["pins"]["accepted_safe"][0]).read_bytes())
            reference = deepcopy(candidate)
            reference["ndof"] = 426627
            reference["configuration"]["mesh"]["mesh_size_nm"] = 80
            inputs = {name: comparison.artifact(root / relative, root) for name,(relative,_) in fixture["pins"].items()}
            for field,value in (("safe_modes", {**safe["safe_modes"], "sha256":"0"*64}), ("phase_adjustment_applied", True)):
                altered = deepcopy(candidate)
                altered["checkpointed_scattering"][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "SAFE columns"):
                    comparison.validate_shared_basis(altered, reference, safe, fixture["point"], inputs)

    def test_anchor_relation_cannot_infer_identity_from_equal_root_numbers(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = make_fixture(root)
            path = root / fixture["pins"]["accepted_anchor_relation"][0]
            proof = json.loads(path.read_bytes())
            proof["matched_channels"]["reflection"]["candidate_gauge_phase_rad"][0] = .1
            path.write_text(json.dumps(proof), encoding="utf-8")
            fixture["pins"]["accepted_anchor_relation"] = path.relative_to(root).as_posix(), comparison.sha(path)
            with self.assertRaisesRegex(ValueError, "root/gauge"):
                self.build(root, fixture)

    def test_bounded_inputs_refuse_arrays_and_large_metadata(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            array = root / "data.npz"
            array.write_bytes(b"synthetic")
            with self.assertRaisesRegex(ValueError, "bounded"):
                comparison.checked_file(array, comparison.sha(array))
            metadata = root / "large.json"
            metadata.write_bytes(b" " * (4*2**20+1))
            with self.assertRaisesRegex(ValueError, "bounded"):
                comparison.checked_file(metadata, comparison.sha(metadata))

    def test_immutable_publication_preserves_bytes_times_and_refuses_conflict(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            value = self.build(root, make_fixture(root))
            output = root / "output"
            path = comparison.write_outputs(output, value)
            before = {file.name:(file.read_bytes(),file.stat().st_mtime_ns) for file in output.iterdir()}
            comparison.write_outputs(output, value)
            self.assertEqual({file.name:(file.read_bytes(),file.stat().st_mtime_ns) for file in output.iterdir()}, before)
            altered = deepcopy(value)
            altered["power_passed"] = False
            with self.assertRaisesRegex(ValueError, "Existing scientific output differs"):
                comparison.write_outputs(output, altered)
            self.assertEqual({file.name:(file.read_bytes(),file.stat().st_mtime_ns) for file in output.iterdir()}, before)
            with (output/"frequency_inplane_comparison_AP_pnc.csv").open(newline="",encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),50)
            self.assertIn("undefined", (output/"report_AP_pnc.html").read_text(encoding="utf-8"))
            self.assertEqual(path.name, "frequency_inplane_comparison_AP_pnc.json")

    def test_cli_returns_success_for_completed_failed_observations(self):
        with patch.object(comparison, "build_comparison", return_value={"status":"compared_finite_inplane_increment", "power_passed":False}), patch.object(comparison,"write_outputs",return_value=Path("synthetic.json")):
            self.assertEqual(comparison.main(["--candidate","candidate","--point-replay","point.json","--point-replay-sha256","0"*64,"--replay-source-sha256","1"*64,"--output","output"]),0)
