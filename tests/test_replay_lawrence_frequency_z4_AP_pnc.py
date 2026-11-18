"""Independent finite-frequency z4 replay safeguards. Athul Prem | _AP_pnc."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix
from skfem import Basis, ElementTriP2, MeshTri

from tools import replay_lawrence_device_point_AP_pnc as shared
from tools import replay_lawrence_frequency_z4_AP_pnc as replay


def pencil(evanescent=1):
    """Five propagating scalar channels plus one evanescent pair, exactly."""
    omega = 2 * np.pi * replay.FREQUENCY
    wave = np.r_[np.arange(1., 6.), np.full(evanescent, 2j)]
    size = len(wave)
    displacement = np.diag(np.sqrt(2 / (omega * abs(wave))))
    U = np.tile(displacement, (1, 2)).astype(np.complex128)
    roots = np.r_[wave, -wave].astype(np.complex128)
    one_side = np.r_[np.ones(5, bool), np.zeros(evanescent, bool)]
    opened = np.tile(one_side, 2)
    modes = {"wavenumbers_rad_m": roots, "displacements": U, "axial_forces": 1j * roots * U,
             "positive_indices": np.arange(size), "negative_indices": np.arange(size, 2 * size), "propagating": opened,
             "powers_w": np.r_[one_side.astype(float), -one_side.astype(float)], "relative_residuals": np.zeros(2 * size),
             "frequency_hz": np.array(replay.FREQUENCY)}
    mass = np.eye(size) * 1e-20
    section = {"coordinates_yz_m": np.zeros((size, 2)), "K0": omega**2 * mass - np.diag(wave**2).real,
               "mass_kg_per_m": mass, "Q": np.eye(size), "G": np.zeros((size, size))}
    return modes, section


def section_fields(refined=False):
    mesh = MeshTri(np.array([[-1., 1., -1., 1.], [-1., -1., 1., 1.]]), np.array([[0, 0], [1, 3], [3, 2]]))
    if refined:
        mesh = mesh.refined()
    basis = Basis(mesh, ElementTriP2())
    y, z = basis.doflocs
    fields = np.zeros((3, basis.N, 5), complex)
    fields[0, :, 0], fields[1, :, 1], fields[2, :, 2] = 1, 1, 1
    fields[1, :, 3], fields[2, :, 3], fields[0, :, 4] = -z, y, y
    return {"vertices_yz_m": mesh.p.T, "triangles": mesh.t.T,
            "coordinates_yz_m": np.tile(basis.doflocs.T, (3, 1)),
            "component_dofs_xyz": np.arange(3 * basis.N).reshape(3, basis.N),
            "displacements": fields.reshape(3 * basis.N, 5), "density_kg_m3": 5323.}


def relation_edge():
    source, target = section_fields(), section_fields(True)
    permutation = np.array([3, 0, 4, 2, 1])
    gauge = np.exp(1j * np.array([.2, 1.1, -2., .7, -.3]))
    target["displacements"] = target["displacements"][:, permutation] * gauge
    inverse = np.argsort(permutation)
    reference, candidate, groups = {}, {}, {}
    for name, offset, sign in (("incoming", 0, 1), ("reflection", 5, -1)):
        roots = list(range(offset, offset + 5))
        other_roots = list(range(offset + 100, offset + 105))
        first_wave = sign * np.arange(1., 6.)
        second_wave = first_wave[permutation] * 1.1
        reference[name] = {"roots": roots, "k": first_wave, "section": source}
        candidate[name] = {"roots": other_roots, "k": second_wave, "section": target}
        cross = np.eye(5)[:, permutation] * gauge
        groups[name] = {"status": "matched", "reference_roots": roots, "candidate_roots_unordered": other_roots,
            "candidate_roots": np.asarray(other_roots)[inverse].tolist(), "candidate_order": inverse.tolist(),
            "full_normalized_complex_overlap": np.stack((cross.real, cross.imag), axis=-1).tolist(),
            "full_mass_overlap_matrix": abs(cross).tolist(), "mass_overlaps": [1.] * 5,
            "candidate_gauge_complex": np.stack((gauge[inverse].real, gauge[inverse].imag), axis=-1).tolist(),
            "candidate_gauge_phase_rad": np.angle(gauge[inverse]).tolist(),
            "reference_wavenumber_rad_m": first_wave.tolist(), "candidate_wavenumber_rad_m": second_wave[inverse].tolist()}
    reference["reverse_partners"] = dict(zip(range(5), range(5, 10), strict=True))
    candidate["reverse_partners"] = dict(zip(range(100, 105), range(105, 110), strict=True))
    groups["transmission"] = deepcopy(groups["incoming"])
    published = {"status": "matched", "matched_channels": groups, "reverse_partner_consistency": {
        "status": "consistent", "reference_forward_to_reverse": reference["reverse_partners"],
        "candidate_forward_to_reverse": candidate["reverse_partners"]}}
    return published, reference, candidate


def archive_fixture():
    link = {"path": "saved.npz", "sha256": "a" * 64, "bytes": 24}
    config = {"geometry": {"thickness_nm": 200}, "mesh": {"thickness_layers": 4}, "material": {"density_kg_m3": 5323},
              "analysis": {"kind": "scattering", "boundary": "matched_ports", "frequency_GHz": 1, "incident_side": "left"}}
    source = {"source_sha256": {replay.old_z4.COORDINATOR: replay.old_z4.COORDINATOR_PAIR[0], "kernel": "b" * 64},
              "packages": {"numpy": "fixture"}}
    original = {"mesh": link, "mesh_checkpoint_summary": link, "port_modes": link, "K": link, "M": link, "source": source}
    current = deepcopy(source)
    current["source_sha256"][replay.old_z4.COORDINATOR] = replay.old_z4.COORDINATOR_PAIR[1]
    current["platform"] = "Linux-fixture"
    record = {"driver": {"sha256": "c" * 64}, "frequency_hz": replay.FREQUENCY, "displacement_dofs": replay.DOFS,
              "body_stage": {"sha256": replay.BODY_SHA}, "trace_sha256": replay.TRACE_SHA,
              "reference_planes_m": list(shared.PLANES), "mesh": link, "mesh_summary": link,
              "archived_safe_modes": link, "array_files": {"K": link, "M": link}, "safe_section": {"sha256": replay.SECTION_SHA},
              "section_receipt": {"sha256": replay.SECTION_RECEIPT_SHA}, "implementation": current,
              "configuration": {**config, "analysis": {**config["analysis"], "frequency_GHz": replay.FREQUENCY / 1e9},
                                "resources": {"max_dofs": replay.DOFS, "max_port_dofs": replay.PORT_DOFS, "linear_solver": "direct"}},
              "body_assembly_source": source, "slurm_job_id": "123", "numerical_thread_environment": replay.THREADS,
              **dict.fromkeys(("mesh_generated", "body_assembly_performed", "section_assembly_performed", "factorization_performed"), False)}
    return record, original, config


class FrequencyZ4ReplayTests(TestCase):
    def test_complete_root_traction_flux_and_evanescent_replay(self):
        modes, section = pencil()
        incident, returning, metrics = replay.replay_modes(modes, section)
        self.assertEqual((incident, returning), (list(range(5)), list(range(6, 11))))
        self.assertEqual(metrics["roots_checked"], 12)
        self.assertLess(metrics["maximum_original_backward_residual"], 1e-15)
        for mutate in (lambda value: value["axial_forces"].__setitem__((0, 0), value["axial_forces"][0, 0] * 1.0001),
                       lambda value: value["positive_indices"].__setitem__(5, 11),
                       lambda value: value["powers_w"].__setitem__(5, .1),
                       lambda value: value["displacements"].__setitem__((0, 0), np.nan)):
            wrong = deepcopy(modes)
            mutate(wrong)
            with self.assertRaises(ValueError):
                replay.replay_modes(wrong, section)
        wrong = deepcopy(section)
        wrong["K0"][0, 0] *= 1.001
        with self.assertRaisesRegex(ValueError, "eigenpair backward residual"):
            replay.replay_modes(modes, wrong)

    def test_collinear_evanescent_basis_and_missing_full_root_partner_are_rejected(self):
        modes, section = pencil(evanescent=2)
        replay.replay_modes(modes, section)
        wrong = deepcopy(modes)
        for key in ("displacements", "axial_forces"):
            wrong[key][:, 5] = wrong[key][:, 6]
        with self.assertRaisesRegex(ValueError, "basis condition"):
            replay.replay_modes(wrong, section)
        with self.assertRaisesRegex(ValueError, "pairing"):
            replay.safe_replay.pair_roots(np.array([1., -1., 2j, -3j]), 1e-6)

    def test_private_frequency_gate_preserves_historical_constants(self):
        before = shared.FREQUENCY, shared.OMEGA, shared.INCIDENT, shared.TRACE_SHA
        core = replay.frequency_core(range(5), range(5, 10))
        self.assertEqual(core.FREQUENCY, replay.FREQUENCY)
        self.assertEqual(core.DOFS, replay.DOFS)
        self.assertEqual(before, (shared.FREQUENCY, shared.OMEGA, shared.INCIDENT, shared.TRACE_SHA))
        matrix = csc_matrix(np.array([[2 + 1j, -1j], [3j, 4]]))
        vector = np.array([[1 + 2j], [3 - 1j]])
        rhs = matrix @ vector
        core.check_original_equations(matrix, rhs, vector)
        with self.assertRaisesRegex(ValueError, "Original equation replay failed"):
            core.check_original_equations(matrix, rhs, vector * (1 + 2e-10))

    def test_five_drive_physical_and_coherent_flux_uses_private_8875_omega(self):
        modes, _ = pencil()
        incident, returning = tuple(range(5)), tuple(range(6, 11))
        core = replay.frequency_core(incident, returning)
        U, F = modes["displacements"], modes["axial_forces"]
        reflection = np.diag(np.arange(1., 6.) / 10)
        transmission = np.diag(np.sqrt(1 - np.diag(reflection)**2) * np.exp(1j * np.arange(5) / 4))
        vectors = np.vstack((U[:, incident] + U[:, returning] @ reflection, U[:, incident] @ transmission))
        force = np.vstack((-F[:, incident] - F[:, returning] @ reflection, F[:, incident] @ transmission))
        rhs = np.vstack((-2 * F[:, incident], np.zeros((6, 5))))
        fields = {"incident_powers_w": np.ones(5)}
        for name, roots, block in (("reflection", np.arange(6, 12), reflection), ("transmission", np.arange(6), transmission)):
            fields[f"{name}_all_mode_indices"] = roots
            fields[f"{name}_amplitudes"] = block
            fields[f"{name}_amplitudes_full"] = np.vstack((block, np.zeros((1, 5))))
        result = core.replay_ports(modes, fields, np.arange(6), np.arange(6, 12), vectors, force, rhs,
                                   np.linalg.norm(rhs, axis=0))
        np.testing.assert_allclose(result[1][0], np.diag(reflection)**2, atol=1e-15)
        self.assertLess(max(result[2]), 1e-14)
        self.assertLess(result[-1]["coherent_coefficient_power_defect_spectral_norm"], 1e-14)
        self.assertEqual(shared.FREQUENCY, 100_000_000)

    def test_p2_richer_trace_assignment_and_each_direct_gauge(self):
        published, reference, candidate = relation_edge()
        mapping, metrics = replay.check_edge(published, reference, candidate)
        self.assertEqual(len(mapping["incoming"]), 5)
        self.assertGreater(metrics["reflection"]["minimum_mass_overlap"], .999999)
        wrong = deepcopy(published)
        wrong["matched_channels"]["incoming"]["candidate_gauge_complex"][0] = [1., 0.]
        with self.assertRaisesRegex(ValueError, "direct edge gauge"):
            replay.check_edge(wrong, reference, candidate)
        wrong = deepcopy(published)
        wrong["matched_channels"]["incoming"]["candidate_roots"] = list(reversed(wrong["matched_channels"]["incoming"]["candidate_roots"]))
        with self.assertRaisesRegex(ValueError, "assigned roots"):
            replay.check_edge(wrong, reference, candidate)

    def test_ambiguous_modes_and_spoofed_reverse_partners_stop_relation(self):
        published, reference, candidate = relation_edge()
        wrong = deepcopy(candidate)
        for name in ("incoming", "reflection"):
            wrong[name]["section"]["displacements"][:, 0] = wrong[name]["section"]["displacements"][:, 1]
        with self.assertRaisesRegex(ValueError, "unresolved"):
            replay.check_edge(published, reference, wrong)
        wrong = deepcopy(published)
        wrong["reverse_partner_consistency"]["candidate_forward_to_reverse"][100] = 109
        with self.assertRaisesRegex(ValueError, "reverse-partner map"):
            replay.check_edge(wrong, reference, candidate)
        wrong = deepcopy(published)
        wrong["status"] = "unresolved"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            replay.check_edge(wrong, reference, candidate)

    def test_assigned_root_square_checks_identity_but_no_gauge_holonomy(self):
        direct = {name: dict(zip(range(5), range(10, 15), strict=True)) for name in ("incoming", "reflection")}
        cross = {name: dict(zip(range(10, 15), range(30, 35), strict=True)) for name in direct}
        old = {"matched_channels": {name: {"reference_roots": list(range(5)), "candidate_roots": list(range(20, 25)),
                "candidate_gauge_phase_rad": [1.] * 5} for name in direct}}
        frequency = {"matched_channels": {name: {"reference_roots": list(range(20, 25)), "candidate_roots": list(range(30, 35)),
                      "candidate_gauge_phase_rad": [2.] * 5} for name in direct}}
        self.assertTrue(replay.check_root_square(direct, cross, old, frequency))
        wrong = deepcopy(cross)
        wrong["reflection"][10], wrong["reflection"][11] = wrong["reflection"][11], wrong["reflection"][10]
        with self.assertRaisesRegex(ValueError, "square is inconsistent"):
            replay.check_root_square(direct, wrong, old, frequency)

    def test_archived_physics_scope_and_external_source_pins_are_strict(self):
        record, original, config = archive_fixture()
        replay.check_archive_identity(record, original, config, "c" * 64)
        for mutate in (lambda value: value["configuration"]["geometry"].__setitem__("thickness_nm", 201),
                       lambda value: value.__setitem__("body_assembly_performed", True),
                       lambda value: value["driver"].__setitem__("sha256", "d" * 64),
                       lambda value: value.__setitem__("frequency_hz", 1_000_000_000),
                       lambda value: value.__setitem__("numerical_thread_environment", {})):
            wrong = deepcopy(record)
            mutate(wrong)
            with self.assertRaises(ValueError):
                replay.check_archive_identity(wrong, original, config, "c" * 64)

    def test_prebody_receipt_cannot_spoof_roots_sources_or_finite_scope(self):
        link = {"path": "safe.json", "bytes": 20, "sha256": "a" * 64}
        modes = {**link, "path": "modes.npz"}
        relation = {**link, "path": "relation.json"}
        safe = {"incident_port_roots": list(range(5)), "backward_port_roots": list(range(5, 10)), "safe_modes": modes}
        receipt = {"status": "independently_verified_safe", "relation_status": "matched", "frequency_hz": replay.FREQUENCY,
                   "problem_dofs": replay.DOFS, "port_dofs": replay.PORT_DOFS, "all_discrete_roots_replayed": 1350,
                   "positive_open_channels": 5, "negative_open_channels": 5, **safe, "safe_receipt": link, "relation": relation,
                   "assigned_root_square_consistent": True, "implementation": {"verifier": replay.artifact(Path(replay.__file__))}, **replay.SCOPE}
        replay.check_prebody_verification(receipt, safe, link, relation)
        for mutate in (lambda value: value.__setitem__("all_discrete_roots_replayed", 10),
                       lambda value: value.__setitem__("relation_status", "unresolved"),
                       lambda value: value.__setitem__("spatial_convergence_established", True),
                       lambda value: value["safe_receipt"].__setitem__("sha256", "b" * 64),
                       lambda value: value["implementation"]["verifier"].__setitem__("sha256", "b" * 64)):
            wrong = deepcopy(receipt)
            mutate(wrong)
            with self.assertRaises(ValueError):
                replay.check_prebody_verification(wrong, safe, link, relation)

    def test_exact_saved_prebody_producer_requires_immutable_receipt_and_snapshot(self):
        link = {"path": "safe.json", "bytes": 20, "sha256": "a" * 64}
        modes, relation = {**link, "path": "modes.npz"}, {**link, "path": "relation.json"}
        safe = {"incident_port_roots": list(range(5)), "backward_port_roots": list(range(5, 10)), "safe_modes": modes}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "retained_prebody_producer.py"
            snapshot.write_bytes(b"#" + b" " * 40610)
            source_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            producer = {"path": "/original/tools/replay_lawrence_frequency_z4_AP_pnc.py", "bytes": 40611, "sha256": source_sha}
            receipt = {"status": "independently_verified_safe", "relation_status": "matched", "frequency_hz": replay.FREQUENCY,
                "problem_dofs": replay.DOFS, "port_dofs": replay.PORT_DOFS, "all_discrete_roots_replayed": 1350,
                "positive_open_channels": 5, "negative_open_channels": 5, **safe, "safe_receipt": link, "relation": relation,
                "assigned_root_square_consistent": True, "implementation": {"verifier": producer}, **replay.SCOPE}
            path = root / "accepted_prebody.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with patch.multiple(replay, ACCEPTED_PREBODY_PRODUCER=snapshot,
                                ACCEPTED_PREBODY_PRODUCER_SHA=source_sha, ACCEPTED_PREBODY_SHA=replay.shared.file_sha(path)):
                checked = replay.check_prebody_verification(receipt, safe, link, relation, path)
                self.assertEqual(checked["status"], "verified_exact_saved_prebody_producer")
                self.assertEqual(checked["retained_producer"]["sha256"], source_sha)
                self.assertEqual(checked["consumer"], replay.artifact(Path(replay.__file__)))
                wrong = deepcopy(receipt)
                wrong["implementation"]["verifier"]["sha256"] = "b" * 64
                with self.assertRaisesRegex(ValueError, "exact accepted receipt"):
                    replay.check_prebody_verification(wrong, safe, link, relation, path)
                with self.assertRaisesRegex(ValueError, "exact accepted receipt"):
                    replay.check_prebody_verification(receipt, safe, link, relation)
                snapshot.write_bytes(snapshot.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "exact accepted receipt"):
                    replay.check_prebody_verification(receipt, safe, link, relation, path)
                snapshot.write_bytes(b"#" + b" " * 40610)
                path.write_text(json.dumps({**receipt, "elapsed_seconds": 0}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exact accepted receipt"):
                    replay.check_prebody_verification(receipt, safe, link, relation, path)

    def test_actual_export_schema_requires_linked_explicit_false_phase_evidence(self):
        runner_sha = replay.FROZEN_RUNNER_SHA
        common = {"configuration": {}, "safe_modes": {}, "safe_section": {}, "section_receipt": {},
                  "incident_port_roots": list(range(5)), "backward_port_roots": list(range(5, 10)),
                  "safe_section_array_sha256": "section", "frequency_hz": replay.FREQUENCY, "trace_sha256": replay.TRACE_SHA,
                  "implementation": {}, "reviewed_sources": {}, "superlu_library": {}, "driver": {"sha256": runner_sha},
                  "hostname": "fixture", "slurm_job_id": "123", "numerical_thread_environment": replay.THREADS,
                  "numerical_backend": {}}
        safe = {**common, "status": "safe_ready"}
        stage_link = {"path": "/stage/stage.json", "bytes": 20, "sha256": "a" * 64}
        solve_link = {"path": "/solve/solve.json", "bytes": 21, "sha256": "b" * 64}
        hashes = {"operator_csc_sha256": "operator", "rhs_array_sha256": "rhs", "operator_nonzeros": 1}
        report = {**hashes, "relative_target": 1e-10, "relative_residual_limit": 1e-9, "backward_residual_limit": 1e-9,
                  "factor": {"permc_spec": "COLAMD", "factor_nonzeros": 2 * replay.DOFS},
                  "channels": [{"incident_port_root": root} for root in range(5)]}
        stage = {**common, **hashes, "status": "operator_staged", "reference_planes_m": list(replay.shared.PLANES),
                 **dict.fromkeys(("safe_eigensolve_performed", "phase_adjustment_applied", "spatial_convergence_established",
                                  "frequency_convergence_established"), False)}
        solved = {**common, "status": "drives_accepted", "stage": stage_link, "linear_solver": report}
        exported = {**common, "status": "completed_unverified", "stage": stage_link, "solve": solve_link,
                    "spatial_convergence_established": False, "frequency_convergence_established": False}
        provenance = {"phase_adjustment_applied": False, "reference_planes_m": list(replay.shared.PLANES),
                      "prepared_frequency_stage": stage_link, "solve_receipt": solve_link,
                      **{name: {key: phase[key] for key in replay.EXECUTION_KEYS} for name, phase in (
                          ("frequency_equation_preparation", stage), ("linux_solve", solved), ("linux_export", exported))}}
        summary = {"checkpointed_scattering": provenance}
        self.assertEqual(replay.check_phases(stage, safe, solved, exported, runner_sha), report)
        evidence = replay.check_publication_phase(summary, stage, solved, exported)
        self.assertFalse(evidence["export_receipt_has_phase_flag"])
        for invalid in (True, None, 0):
            with self.assertRaisesRegex(ValueError, "finite scope"):
                replay.check_phases(stage, safe, solved, {**exported, "phase_adjustment_applied": invalid}, runner_sha)
        for invalid in (True, None, 0):
            wrong = deepcopy(summary)
            wrong["checkpointed_scattering"]["phase_adjustment_applied"] = invalid
            with self.assertRaisesRegex(ValueError, "public phase flag"):
                replay.check_publication_phase(wrong, stage, solved, exported)
        wrong = deepcopy(summary)
        del wrong["checkpointed_scattering"]["phase_adjustment_applied"]
        with self.assertRaises(KeyError):
            replay.check_publication_phase(wrong, stage, solved, exported)
        wrong = deepcopy(summary)
        wrong["checkpointed_scattering"]["prepared_frequency_stage"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "execution provenance"):
            replay.check_publication_phase(wrong, stage, solved, exported)
        wrong = deepcopy(summary)
        wrong["checkpointed_scattering"]["linux_export"]["slurm_job_id"] = "other"
        with self.assertRaisesRegex(ValueError, "execution provenance"):
            replay.check_publication_phase(wrong, stage, solved, exported)
        for phase in (safe, stage, solved, exported):
            phase["driver"] = {"sha256": "c" * 64}
        with self.assertRaisesRegex(ValueError, "finite scope"):
            replay.check_phases(stage, safe, solved, exported, "c" * 64)

    def test_artifact_binding_handles_platform_paths_but_preserves_name_bytes_hash(self):
        first = {"path": "C:\\saved\\modes.npz", "sha256": "a" * 64, "bytes": 12}
        second = {"path": "/saved/modes.npz", "sha256": "a" * 64, "bytes": 12}
        self.assertTrue(replay.same_artifact(first, second))
        self.assertFalse(replay.same_artifact(first, {**second, "path": "/saved/wrong.npz"}))
        self.assertFalse(replay.same_artifact(first, {**second, "sha256": "b" * 64}))

    def test_public_csv_joins_all_five_drives_at_8875_and_rejects_spoofs(self):
        incident, returning = list(range(5)), list(range(5, 10))
        reflection = np.eye(5, dtype=complex) * .1
        reflection[0, 1] = .0008 + .0003j
        transmission = np.eye(5, dtype=complex) * .9j
        blocks = {name: (value, abs(value)**2) for name, value in (("reflection", reflection), ("transmission", transmission))}
        r, t = (blocks[name][1].sum(axis=0) for name in ("reflection", "transmission"))
        totals = r, t, r, t, np.zeros(5)
        coefficients = [{"kind": kind, "incident_port_index": incoming, "outgoing_port_index": outgoing,
            "incident_mode_order": column + 1, "amplitude_real": value[row, column].real,
            "amplitude_imag": value[row, column].imag, "power_fraction": abs(value[row, column])**2}
            for kind, roots, value in (("reflection", returning, reflection), ("transmission", incident, transmission))
            for column, incoming in enumerate(incident) for row, outgoing in enumerate(roots)]
        total_rows = [{"incident_mode_order": i + 1, "frequency_GHz": .8875, "reflection": r[i], "transmission": t[i],
            "reflection_flux": r[i], "transmission_flux": t[i], "power_balance": 1 - r[i] - t[i],
            "flux_power_balance": 1 - r[i] - t[i]} for i in range(5)]
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name, rows):
                with (root / name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)

            write("scattering_channels_AP_pnc.csv", coefficients)
            write("scattering_AP_pnc.csv", total_rows)
            replay.compare_public_tables(root, blocks, totals, incident, returning)
            wrong = deepcopy(total_rows)
            wrong[0]["frequency_GHz"] = .1
            write("scattering_AP_pnc.csv", wrong)
            with self.assertRaisesRegex(ValueError, "frequency"):
                replay.compare_public_tables(root, blocks, totals, incident, returning)
            write("scattering_AP_pnc.csv", total_rows)
            wrong = deepcopy(coefficients)
            wrong[5]["amplitude_real"] += .0001
            write("scattering_channels_AP_pnc.csv", wrong)
            with self.assertRaisesRegex(ValueError, "complex coefficient"):
                replay.compare_public_tables(root, blocks, totals, incident, returning)
            wrong = deepcopy(coefficients)
            wrong[-1] = wrong[0]
            write("scattering_channels_AP_pnc.csv", wrong)
            with self.assertRaisesRegex(ValueError, "duplicated"):
                replay.compare_public_tables(root, blocks, totals, incident, returning)
