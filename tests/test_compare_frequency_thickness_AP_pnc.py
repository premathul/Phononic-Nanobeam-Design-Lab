"""Synthetic finite-increment and signed-relation checks. Athul Prem | _AP_pnc."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from tools import compare_frequency_thickness_AP_pnc as comparison


def signed_modes(roots, reverse, permutation=None, forward_phase=None, reverse_phase=None):
    permutation = np.arange(5) if permutation is None else np.asarray(permutation)
    forward_phase = np.ones(5) if forward_phase is None else np.asarray(forward_phase)
    reverse_phase = np.ones(5) if reverse_phase is None else np.asarray(reverse_phase)
    values = np.vstack((np.eye(5), np.zeros((10, 5)))).astype(complex)
    waves = np.arange(1., 6.)[permutation]
    return {"incoming": {"roots": np.asarray(roots), "k": waves,
                         "displacements": values[:, permutation] * forward_phase},
            "reflection": {"roots": np.asarray(reverse), "k": -waves,
                           "displacements": values[:, permutation] * reverse_phase},
            "reverse_partners": dict(zip(roots, reverse, strict=True))}


def synthetic_section():
    basis = SimpleNamespace(N=5, doflocs=np.vstack((np.arange(5.), np.zeros(5))),
                            mesh=SimpleNamespace(t=np.array([[0], [1], [2]])))
    return comparison.matching.Section(basis, np.arange(15).reshape(3, 5), np.eye(5), "synthetic")


def map_edge(incoming, reflection):
    return {"status": "matched", "matched_channels": {
        name: {"reference_roots": list(mapping), "candidate_roots": list(mapping.values())}
        for name, mapping in (("incoming", incoming), ("reflection", reflection))}}


class FrequencyThicknessTests(TestCase):
    def test_unequal_p2_traces_match_the_same_physical_vector_fields(self):
        from skfem.models.poisson import mass
        def fixture(z_nodes):
            mesh = comparison.matching.MeshTri.init_tensor(np.linspace(-.5, .5, 3), np.linspace(-.1, .1, z_nodes))
            basis = comparison.matching.Basis(mesh, comparison.matching.ElementTriP2(), intorder=4)
            section = comparison.matching.Section(basis, np.arange(3 * basis.N).reshape(3, basis.N),
                                                  mass.assemble(basis).toarray(), "synthetic")
            fields = np.zeros((3, basis.N, 5), dtype=complex)
            fields[0, :, 0] = fields[1, :, 1] = fields[2, :, 2] = 1
            fields[1, :, 3], fields[2, :, 4] = basis.doflocs[1], basis.doflocs[0]
            modes = signed_modes([1, 2, 3, 4, 5], [11, 12, 13, 14, 15])
            for name in ("incoming", "reflection"):
                modes[name]["displacements"] = fields.reshape(3 * basis.N, 5).copy()
            return section, modes
        first_section, first = fixture(3)
        second_section, second = fixture(4)
        self.assertNotEqual(first_section.basis.N, second_section.basis.N)
        phase = np.exp(1j * np.array([.1, .4, -.3, 1.2, -2.1]))
        for name in ("incoming", "reflection"):
            second[name]["displacements"] *= phase
        result = comparison.match_edge(first_section, first, second_section, second)
        self.assertEqual(result["status"], "matched")
        for name in ("incoming", "reflection"):
            np.testing.assert_allclose(result["matched_channels"][name]["mass_overlaps"], 1, atol=2e-15)
            np.testing.assert_allclose(np.exp(1j * np.asarray(result["matched_channels"][name]["candidate_gauge_phase_rad"])), phase, atol=2e-15)

    def test_signed_matching_recovers_permutations_and_independent_gauges(self):
        roots, reverse = [11, 4, 27, 9, 31], [21, 14, 37, 19, 41]
        candidate, candidate_reverse = [118, 102, 109, 140, 113], [218, 202, 209, 240, 213]
        permutation = np.array([2, 4, 1, 0, 3])
        phase_in = np.exp(1j * np.array([.3, -.7, 1.8, -.9, 2.1]))
        phase_out = np.exp(1j * np.array([-.1, 1.7, -.8, .6, -2.4]))
        first = signed_modes(roots, reverse)
        second = signed_modes(candidate, candidate_reverse, permutation, phase_in, phase_out)
        section = synthetic_section()
        result = comparison.match_edge(section, first, section, second)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["reverse_partner_consistency"]["status"], "consistent")
        inverse = np.argsort(permutation)
        for name, expected in (("incoming", phase_in), ("reflection", phase_out)):
            row = result["matched_channels"][name]
            self.assertEqual(row["candidate_order"], inverse.tolist())
            np.testing.assert_allclose(np.exp(1j * np.asarray(row["candidate_gauge_phase_rad"])), expected[inverse], atol=1e-15)
            self.assertEqual(np.asarray(row["full_normalized_complex_overlap"]).shape, (5, 5, 2))
        changed = deepcopy(second)
        changed["incoming"]["displacements"][:, 0] = changed["incoming"]["displacements"][:, 1]
        self.assertEqual(comparison.match_edge(section, first, section, changed)["status"], "unresolved")

    def test_root_square_checks_identities_without_imposing_gauge_closure(self):
        old_cross = map_edge({1: 11, 2: 12}, {3: 13, 4: 14})
        old_frequency = map_edge({11: 111, 12: 112}, {13: 113, 14: 114})
        new_frequency = map_edge({1: 21, 2: 22}, {3: 23, 4: 24})
        new_cross = map_edge({21: 111, 22: 112}, {23: 113, 24: 114})
        for edge in (old_cross, old_frequency, new_frequency, new_cross):
            for group in edge["matched_channels"].values():
                group["candidate_gauge_phase_rad"] = [.7, -.2]
        square = comparison.permutation_square(old_cross, old_frequency, new_frequency, new_cross)
        self.assertEqual(square["status"], "consistent")
        self.assertIs(square["gauge_closure_required"], False)
        wrong = deepcopy(new_cross)
        wrong["matched_channels"]["reflection"]["candidate_roots"].reverse()
        self.assertEqual(comparison.permutation_square(old_cross, old_frequency, new_frequency, wrong)["status"], "unresolved")

    def test_weak_pair_failure_survives_a_passing_absolute_gate(self):
        result = comparison.pair_targets([8.4e-7, 9.0e-7], [8e-7, 8e-7], [1e-3] * 2, [1e-3] * 2)
        self.assertIs(result["directed_increment_passed"], False)
        self.assertTrue(result["directed_increment_rows"][0]["passed"])
        self.assertFalse(result["directed_increment_rows"][1]["passed"])
        self.assertTrue(result["suppression_retention_passed"])
        self.assertLess(max(abs(a - 8e-7) for a in (8.4e-7, 9e-7)), comparison.matching.LIMIT)
        self.assertIs(result["denominator_floor_applied"], False)
        # Exact 10% and 1% boundaries are inclusive; no hidden tolerance is added.
        edge = comparison.pair_targets([11., 10.], [10., 10.], [1100., 1000.], [1000., 1000.])
        self.assertTrue(edge["directed_increment_passed"])
        self.assertTrue(edge["suppression_retention_passed"])

    def test_all_four_suppression_rows_and_exact_positive_denominators_are_required(self):
        for index in range(4):
            anchors = [[1e-3, 1e-3], [1e-3, 1e-3]]
            anchors[index // 2][index % 2] = 5e-5
            result = comparison.pair_targets([8e-7] * 2, [8e-7] * 2, *anchors)
            self.assertFalse(result["suppression_retention_passed"])
            self.assertEqual(sum(not row["passed"] for row in result["suppression_retention_rows"]), 1)
        for denominator in (0., -1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "no floor"):
                comparison.pair_targets([8e-7] * 2, [denominator, 8e-7], [1e-3] * 2, [1e-3] * 2)
        tiny = comparison.pair_targets([1e-30] * 2, [1e-30] * 2, [1e-20] * 2, [1e-20] * 2)
        self.assertEqual(tiny["directed_increment_rows"][0]["exact_denominator"], 1e-30)

    def test_raw_and_aligned_complex_changes_keep_zero_cases_explicit(self):
        reference, phase = .0003 + .0008j, np.exp(.7j)
        result = comparison.complex_pair_change(reference, reference * phase, reference)
        self.assertAlmostEqual(result["raw"]["phase_change_rad"], .7)
        self.assertGreater(result["raw"]["relative_absolute_complex_change"], .1)
        self.assertEqual(result["gauge_aligned"]["absolute_complex_change"], 0)
        self.assertIs(result["additional_phase_gate_applied"], False)
        zero = comparison.complex_pair_change(0j, 1j, 1j)
        self.assertIsNone(zero["raw"]["relative_absolute_complex_change"])
        self.assertIsNone(zero["raw"]["phase_change_rad"])
        self.assertEqual(zero["raw"]["absolute_complex_change"], 1.)

    def test_response_parser_uses_raw_csv_order_instead_of_matched_candidate_order(self):
        roots, reverse = [9, 2, 7, 4, 1], [19, 12, 17, 14, 11]
        summary = {"frequency_hz": comparison.FREQUENCY, "incident_side": "left", "incident_mode_indices": roots,
                   "num_incident_channels": 5, "power_balance_passed": True, "run_hash": "synthetic"}
        totals, coefficients = [], []
        for i, root in enumerate(roots):
            totals.append({"incident_mode_order": i + 1, "frequency_GHz": .8875, "reflection": .01, "transmission": .99})
            for kind, out, amplitude in (("reflection", reverse, .1j), ("transmission", roots, np.sqrt(.99))):
                for j, outgoing in enumerate(out):
                    value = amplitude if i == j else 0j
                    coefficients.append({"incident_mode_order": i + 1, "incident_port_index": root, "outgoing_port_index": outgoing,
                        "kind": kind, "amplitude_real": float(np.real(value)), "amplitude_imag": float(np.imag(value)),
                        "power_fraction": float(abs(value)**2)})
        edge = {"matched_channels": {"incoming": {"candidate_roots": roots[::-1]}},
                "reverse_partner_consistency": {"candidate_forward_to_reverse": dict(zip(map(str, roots), reverse, strict=True))}}
        parsed = comparison.parsed_response(summary, [totals[::-1], coefficients[::-1]], edge, "candidate", comparison.FREQUENCY)
        self.assertEqual([row["root"] for row in parsed["rows"]], roots)
        self.assertEqual(len(parsed["coefficients"]), 50)

    def test_new_safe_receipt_is_bound_to_reviewed_source_and_archived_section(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = root / "tools"
            tools.mkdir()
            runner = tools / "run_lawrence_frequency_z4_AP_pnc.py"
            runner.write_text("reviewed synthetic runner", encoding="utf-8")
            modes = root / "modes-887500000_AP_pnc.npz"
            modes.write_bytes(b"synthetic metadata-only file")
            record = {"status": "safe_ready", "frequency_hz": comparison.FREQUENCY, "displacement_dofs": 426627,
                "discrete_roots": 1350, "positive_open_channels": 5, "negative_open_channels": 5,
                "cutoff_status": "solver_resolved_at_sample", "degeneracy_status": "none_detected",
                "coincident_propagating_root_groups": [], "reference_planes_m": comparison.PLANES,
                "trace_sha256": comparison.TRACE_SHA, "driver": comparison.artifact(runner, root),
                "body_stage": {"sha256": comparison.BODY_STAGE_SHA},
                "safe_section": {"sha256": comparison.PINS["z4_section"][1]},
                "section_receipt": {"sha256": comparison.PINS["z4_section_receipt"][1]},
                "safe_modes": comparison.artifact(modes, root)}
            for name in ("body_assembly_performed", "section_assembly_performed", "mesh_generated",
                         "anchor_eigensolve_performed", "factorization_performed"):
                record[name] = False
            path = root / "safe_AP_pnc.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            actual, actual_modes, _ = comparison.checked_safe(path, comparison.sha(path), comparison.sha(runner), root)
            self.assertEqual(actual["status"], "safe_ready")
            self.assertTrue(actual_modes.samefile(modes))
            record["safe_section"]["sha256"] = "0" * 64
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance"):
                comparison.checked_safe(path, comparison.sha(path), comparison.sha(runner), root)
            with self.assertRaisesRegex(ValueError, "externally accepted"):
                comparison.checked_safe(path, "0" * 64, comparison.sha(runner), root)

    def test_publication_preserves_accepted_bytes_and_rejects_changed_scientific_content(self):
        with TemporaryDirectory() as temporary:
            document = {"phase": "saved_section_relation", "status": "signed_section_relation_accepted",
                        "root_permutation_square": {"status": "consistent"}}
            path = comparison.write_outputs(temporary, document)
            before = path.read_bytes(), path.stat().st_mtime_ns
            comparison.write_outputs(temporary, document)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
            altered = deepcopy(document)
            altered["root_permutation_square"]["status"] = "unresolved"
            with self.assertRaisesRegex(ValueError, "Existing scientific output differs"):
                comparison.write_outputs(temporary, altered)
            old_hash = comparison.sha(path)
            value = json.loads(path.read_bytes())
            value["root_permutation_square"]["status"] = "unresolved"
            value["content_sha256"] = comparison.digest({k: v for k, v in value.items() if k != "content_sha256"})
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "externally accepted"):
                comparison.accepted_relation(path, old_hash)

    def test_new_mode_reader_checks_signed_transport_and_complete_partition(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "modes.npz"
            positive, negative = np.arange(5), np.arange(5, 10)
            u = np.column_stack((np.eye(5), np.eye(5))).astype(np.complex128)
            force = u * np.r_[np.ones(5), -np.ones(5)] * 1j / (np.pi * comparison.FREQUENCY)
            arrays = {"frequency_hz": comparison.FREQUENCY, "wavenumbers_rad_m": np.r_[np.arange(1., 6.), -np.arange(1., 6.)].astype(complex),
                "powers_w": np.r_[np.ones(5), -np.ones(5)], "propagating": np.ones(10, dtype=bool),
                "relative_residuals": np.zeros(10), "positive_indices": positive, "negative_indices": negative,
                "displacements": u, "axial_forces": force}
            np.savez(path, **arrays)
            actual = comparison.read_modes(path, comparison.FREQUENCY, 5, positive, negative)
            self.assertEqual(actual["reverse_partners"], dict(zip(positive, negative, strict=True)))
            arrays["axial_forces"] = -force
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "unit-power"):
                comparison.read_modes(path, comparison.FREQUENCY, 5, positive, negative)
            arrays["axial_forces"] = force
            arrays["negative_indices"] = np.array([4, 6, 7, 8, 9])
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "partition"):
                comparison.read_modes(path, comparison.FREQUENCY, 5, positive, negative)

    def test_only_exact_retained_prebody_receipt_and_source_can_cross_a_consumer_correction(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "tools/replay_lawrence_frequency_z4_AP_pnc.py"
            current.parent.mkdir()
            current.write_text("current reviewed receipt consumer", encoding="utf-8")
            snapshot = root / comparison.PREBODY_PRODUCER_SNAPSHOT
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("exact executed prebody producer", encoding="utf-8")
            old_sha, old_bytes = comparison.sha(snapshot), snapshot.stat().st_size
            producer = {"path": "tools/replay_lawrence_frequency_z4_AP_pnc.py", "sha256": old_sha, "bytes": old_bytes}
            path = root / "prebody.json"
            path.write_text(json.dumps({"implementation": {"verifier": producer}}), encoding="utf-8")
            accepted = comparison.sha(path)
            with patch.multiple(comparison, PREBODY_RECEIPT_SHA=accepted,
                                PREBODY_PRODUCER_SHA=old_sha, PREBODY_PRODUCER_BYTES=old_bytes):
                before = path.read_bytes(), path.stat().st_mtime_ns
                result = comparison.checked_replay(path, accepted, comparison.sha(current), root, "prebody")
                compatibility = result["accepted_producer_compatibility"]
                self.assertEqual(compatibility["executed_producer"], producer)
                self.assertEqual(compatibility["retained_executed_producer"]["sha256"], old_sha)
                self.assertEqual(compatibility["current_consumer"]["sha256"], comparison.sha(current))
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
                with self.assertRaisesRegex(ValueError, "externally accepted"):
                    comparison.checked_replay(path, "0" * 64, comparison.sha(current), root, "prebody")
                with self.assertRaisesRegex(ValueError, "exact retained prebody source"):
                    comparison.checked_replay(path, accepted, comparison.sha(current), root, "point")
                wrong = {"implementation": {"verifier": {**producer, "sha256": "1" * 64}}}
                with patch.object(comparison, "read_json", return_value=wrong):
                    with self.assertRaisesRegex(ValueError, "exact retained prebody source"):
                        comparison.checked_replay(path, accepted, comparison.sha(current), root, "prebody")
                snapshot.write_text("different old producer", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "externally accepted"):
                    comparison.checked_replay(path, accepted, comparison.sha(current), root, "prebody")
            current_record = root / "point.json"
            current_record.write_text(json.dumps({"implementation": {"verifier": comparison.artifact(current, root)}}), encoding="utf-8")
            result = comparison.checked_replay(current_record, comparison.sha(current_record), comparison.sha(current), root)
            self.assertIsNone(result["accepted_producer_compatibility"])

    def test_only_exact_executed_relation_and_retained_producer_survive_consumer_change(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / comparison.RELATION_PRODUCER_SNAPSHOT
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("exact executed relation producer", encoding="utf-8")
            producer = {"path": "tools/compare_frequency_thickness_AP_pnc.py", "sha256": comparison.sha(snapshot),
                        "bytes": snapshot.stat().st_size}
            value = {"producer": producer, "phase": "saved_section_relation", "status": "signed_section_relation_accepted",
                "frequency_hz": comparison.FREQUENCY, "reference_frequency_hz": comparison.ANCHOR_FREQUENCY,
                "positive_open_channels": 5, "negative_open_channels": 5, "overlap_gate": .99,
                "root_permutation_square": {"status": "consistent"}, "inputs": {},
                "edges": {"example": {"status": "matched", "reverse_partner_consistency": {"status": "consistent"}}}}
            value["content_sha256"] = comparison.digest(value)
            path = root / "relation.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            accepted = comparison.sha(path)
            with patch.multiple(comparison, EXECUTED_RELATION_SHA=accepted, RELATION_PRODUCER_SHA=producer["sha256"],
                                RELATION_PRODUCER_BYTES=producer["bytes"], PINS={}):
                before = path.read_bytes(), path.stat().st_mtime_ns
                result = comparison.accepted_relation(path, accepted, root)
                self.assertEqual(result["producer"], producer)
                self.assertEqual(result["accepted_producer_compatibility"]["current_consumer"]["sha256"], comparison.sha(comparison.__file__))
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
                wrong = deepcopy(value)
                wrong["producer"]["sha256"] = "1" * 64
                with patch.object(comparison, "read_json", return_value=wrong):
                    with self.assertRaisesRegex(ValueError, "exact retained executed producer"):
                        comparison.accepted_relation(path, accepted, root)
                with self.assertRaisesRegex(ValueError, "externally accepted"):
                    comparison.accepted_relation(path, "0" * 64, root)
                snapshot.write_text("another relation producer", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "externally accepted"):
                    comparison.accepted_relation(path, accepted, root)

    def test_response_comparison_integrates_exact_anchors_and_replay_gates(self):
        with TemporaryDirectory() as temporary:
            root, section = Path(temporary), synthetic_section()
            old4 = signed_modes(list(range(1, 6)), list(range(21, 26)))
            new4 = signed_modes(list(range(11, 16)), list(range(31, 36)))
            new6 = signed_modes(list(range(111, 116)), list(range(131, 136)))
            cross = comparison.match_edge(section, new4, section, new6)
            freq4 = comparison.match_edge(section, old4, section, new4)
            r = np.zeros((5, 5), dtype=complex)
            r[1, 0] = r[0, 1] = np.sqrt(8e-7) * 1j
            t = np.diag(np.sqrt(1 - np.sum(abs(r)**2, axis=0)))
            power = float(abs(r[1, 0])**2)
            summaries = {}
            def write_response(label, modes, layers, dofs, port):
                directory = root / label
                directory.mkdir()
                roots = modes["incoming"]["roots"].tolist()
                reverse = [modes["reverse_partners"][entry] for entry in roots]
                totals, coefficients = [], []
                for col, incoming in enumerate(roots):
                    totals.append({"incident_mode_order": col + 1, "frequency_GHz": .8875,
                        "reflection": float(np.sum(abs(r[:, col])**2)), "transmission": float(np.sum(abs(t[:, col])**2))})
                    for kind, output, matrix in (("reflection", reverse, r), ("transmission", roots, t)):
                        for row, outgoing in enumerate(output):
                            value = matrix[row, col]
                            coefficients.append({"kind": kind, "incident_mode_order": col + 1, "incident_port_index": incoming,
                                "outgoing_port_index": outgoing, "amplitude_real": value.real, "amplitude_imag": value.imag,
                                "power_fraction": float(abs(value)**2)})
                hashes = {}
                for name, rows in (("scattering_AP_pnc.csv", totals), ("scattering_channels_AP_pnc.csv", coefficients)):
                    with (directory / name).open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    hashes[name] = comparison.sha(directory / name)
                sources = {f"src/phononic/{stem}_AP_pnc.py": "synthetic-source" for stem in (*comparison.matching.CORE, "layered", "assembly_batched")}
                summary = {"status": "completed_unverified", "frequency_hz": comparison.FREQUENCY, "incident_side": "left",
                    "incident_mode_indices": roots, "num_incident_channels": 5, "power_balance_passed": True,
                    "run_hash": label, "ndof": dofs, "port_dofs": port, "hole_count": 26, "output_sha256": hashes,
                    "configuration": {"geometry": {"same": True}, "material": {"same": True},
                        "mesh": {"kind": "layered", "thickness_layers": layers, "mesh_size_nm": 80}},
                    "checkpointed_scattering": {"reference_planes_m": comparison.PLANES, "phase_adjustment_applied": False},
                    "implementation": {"packages": {"numpy": "synthetic"}, "source_sha256": sources}}
                path = directory / "summary_AP_pnc.json"
                path.write_text(json.dumps(summary), encoding="utf-8")
                summaries[label] = path
            write_response("z4", new4, 4, 426627, 675)
            write_response("z6", new6, 6, 616239, 975)
            relation_path, prebody_path, point_path = [root / name for name in ("relation.json", "prebody.json", "point.json")]
            for path in (relation_path, prebody_path, point_path):
                path.write_text("{}", encoding="utf-8")
            relation = {"edges": {"cross_z_8875": cross, "z4_frequency": freq4}, "producer": {"sha256": "comparison"},
                "reviewed_runner": {"sha256": "runner"}, "candidate_safe_receipt": {"path": "safe.json", "bytes": 2, "sha256": "safe"},
                "candidate_safe_modes": {"path": "modes.npz", "bytes": 2, "sha256": "modes"}}
            prebody = {"status": "independently_verified_safe", "relation_status": "matched",
                "relation_replay": {"status": "independently_verified_signed_relation"}, "assigned_root_square_consistent": True,
                "relation": comparison.artifact(relation_path, root), "safe_receipt": relation["candidate_safe_receipt"],
                "safe_modes": relation["candidate_safe_modes"],
                "implementation": {"comparison": relation["producer"], "runner": relation["reviewed_runner"]}}
            point = {"status": "point_verified_within_saved_discretization", "frequency_hz": comparison.FREQUENCY,
                "prebody_replay": comparison.artifact(prebody_path, root), "public_summary_sha256": comparison.sha(summaries["z4"]),
                "public_summary": comparison.artifact(summaries["z4"], root), "relative_target": 1e-10,
                "relative_residual_limit": 1e-9, "backward_residual_limit": 1e-9,
                "maximum_relative_residual": 1e-12, "maximum_backward_residual": 1e-15}
            for flag in ("original_equations_passed", "physical_nodal_force_power_passed", "all_modal_coefficients_passed",
                         "full_p2_field_identity_passed", "coherent_left_incidence_passed"):
                point[flag] = True
            z6_pairs = [{"reference_incident_root": a, "reference_outgoing_root": b, "candidate_incident_root": c,
                         "candidate_outgoing_root": d, "reference_power": .001, "candidate_power": power}
                        for a, b, c, d in ((101, 122, 111, 132), (102, 121, 112, 131))]
            accepted = {"inputs": {"candidate_summary": comparison.artifact(summaries["z6"], root)}, "selected_pair": z6_pairs}
            old = {"selected_conversion_pairs": {name: {"directions": [{"incident_root": a, "outgoing_root": b, "power_fraction": .001}
                    for a, b in pairs]} for name, pairs in (("h80_z4", [(1, 22), (2, 21)]), ("h80_z6", [(101, 122), (102, 121)]))},
                   "preserved_scientific_status": {"complex_passed": False, "mesh_pair_passed": False}}
            paths = {}
            for name, value in (("accepted_z6_frequency_comparison", accepted), ("one_ghz_conversion_verification", old)):
                path = root / (name + ".json")
                path.write_text(json.dumps(value), encoding="utf-8")
                paths[name] = path
            with patch.object(comparison, "load_pins", return_value=(paths, {})), \
                    patch.object(comparison, "accepted_relation", return_value=relation), \
                    patch.object(comparison, "checked_replay", side_effect=lambda path, *args: prebody if path == prebody_path else point):
                def run():
                    return comparison.build_comparison(relation_path, "relation", prebody_path, "prebody", root / "z4",
                        point_path, "point", "source", root)
                result = run()
                self.assertEqual(len(result["coefficients"]), 50)
                self.assertTrue(result["mesh_pair_passed"])
                self.assertTrue(result["pair_specific_targets"]["directed_increment_passed"])
                self.assertTrue(result["pair_specific_targets"]["suppression_retention_passed"])
                self.assertEqual(result["maximum_complex_change"], 0.)
                output = root / "comparison"
                comparison.write_outputs(output, result, root)
                comparison.write_outputs(output, result, root)
                self.assertIn("Raw |ΔS|", (output / "report_AP_pnc.html").read_text(encoding="utf-8"))
                historical = {"accepted_prebody_receipt": comparison.artifact(prebody_path, root),
                    "executed_producer": {"path": "old.py", "sha256": "old", "bytes": 4},
                    "retained_executed_producer": {"path": "retained.py", "sha256": "old", "bytes": 4},
                    "current_consumer": {"path": "current.py", "sha256": "current", "bytes": 7}}
                prebody["accepted_producer_compatibility"] = historical
                point["prebody_producer_compatibility"] = {"status": "verified_exact_saved_prebody_producer",
                    "accepted_prebody_receipt": historical["accepted_prebody_receipt"],
                    "accepted_producer": historical["executed_producer"],
                    "retained_producer": historical["retained_executed_producer"], "consumer": historical["current_consumer"]}
                self.assertEqual(run()["accepted_prebody_producer_compatibility"], historical)
                point["prebody_producer_compatibility"]["retained_producer"] = {"path": "retained.py", "sha256": "different", "bytes": 4}
                with self.assertRaisesRegex(ValueError, "exact accepted prebody producer compatibility"):
                    run()
                point["prebody_producer_compatibility"]["retained_producer"] = historical["retained_executed_producer"]
                point["relative_residual_limit"] = 1e-6
                with self.assertRaisesRegex(ValueError, "accepted original-equation"):
                    run()
