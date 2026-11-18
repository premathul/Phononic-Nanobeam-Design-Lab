"""Five-drive physical replay and producer contract checks. Athul Prem | _AP_pnc."""

from contextlib import nullcontext
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix
from skfem import Basis, ElementTetP2, ElementVector, MeshTet

from tools import replay_lawrence_frequency_h100_AP_pnc as replay
from tools import run_lawrence_frequency_h100_AP_pnc as producer


def small_sources():
    old = {"elastic_kernel": "a" * 64}
    new = dict(old)
    for key, (first, last) in replay.h100.SOURCE_CHANGES.items():
        if first is not None:
            old[key] = first
        new[key] = last
    return ({"source_sha256": old, "packages": {}, "platform": "Windows-fixture"},
            {"source_sha256": new, "packages": {}, "platform": "Linux-fixture"})


def archive_fixture():
    old, new = small_sources()
    link = {"path": "saved.npz", "bytes": 10, "sha256": "a" * 64}
    summary = {**link, "path": "summary_AP_pnc.json", "sha256": replay.MESH_SUMMARY_SHA, "bytes": 9558}
    original = {"source": old, "mesh": link,
                "port_modes": link, "K": link, "M": link}
    config = {"geometry": {"thickness_nm": 200}, "material": {"density_kg_m3": 5323},
              "mesh": {"mesh_size_nm": 100, "thickness_layers": 4},
              "analysis": {"frequency_GHz": 1, "incident_side": "left"},
              "resources": {"assembly_batch_elements": 2048}}
    stage = {"driver": {"sha256": "c" * 64}, "status": "operator_staged", "frequency_hz": replay.FREQUENCY,
             "displacement_dofs": replay.DOFS, "trace_sha256": replay.TRACE_SHA,
             "safe_provider_trace_sha256": replay.z4.TRACE_SHA, "body_stage": {"sha256": replay.BODY_SHA},
             "body_assembly_source": old, "body_source_compatibility": {"historical_execution_source_pairs": {
                 key: list(pair) for key, pair in replay.h100.SOURCE_CHANGES.items()}},
             "mesh": link, "mesh_summary": summary, "archived_safe_modes": link,
             "array_files": {"K": link, "M": link}, "implementation": new,
             "reference_planes_m": list(replay.shared.PLANES),
             "configuration": {**config, "analysis": {**config["analysis"], "frequency_GHz": .8875},
                 "resources": {"max_dofs": replay.DOFS, "max_port_dofs": 675, "linear_solver": "direct", "assembly_batch_elements": 2048}},
             **dict.fromkeys(("mesh_generated", "body_assembly_performed", "section_assembly_performed",
                             "safe_eigensolve_performed", "factorization_performed", "phase_adjustment_applied"), False)}
    return stage, original, config


def phase_fixture():
    link = {"path": "saved.json", "bytes": 10, "sha256": "a" * 64}
    common = {"implementation": {"platform": "Linux-fixture"}, "driver": {"sha256": "c" * 64},
              "reviewed_sources": {}, "superlu_library": {}, "hostname": "fixture", "slurm_job_id": "123",
              "numerical_thread_environment": replay.THREADS, "numerical_backend": {},
              **{key: link for key in replay.CONTEXT_KEYS},
              "reused_execution_helper": {**link, "sha256": replay.EXECUTION_HELPER_SHA}}
    hashes = {"operator_csc_sha256": "operator", "rhs_array_sha256": "rhs", "operator_nonzeros": 10}
    stage = {**deepcopy(common), **hashes, "status": "operator_staged", "configuration": {},
             "reference_planes_m": list(replay.shared.PLANES),
             **dict.fromkeys(("phase_adjustment_applied", "spatial_convergence_established", "frequency_convergence_established"), False)}
    report = {**hashes, "relative_target": 1e-10, "relative_residual_limit": 1e-9, "backward_residual_limit": 1e-9,
              "factor": {"permc_spec": "COLAMD", "factor_nonzeros": 2 * replay.DOFS},
              "channels": [{"incident_port_root": root} for root in replay.INCIDENT]}
    solved = {**deepcopy(common), "status": "drives_accepted", "stage": link, "linear_solver": report}
    exported = {**deepcopy(common), "status": "completed_unverified", "stage": link, "solve": link,
                "reference_planes_m": list(replay.shared.PLANES), "phase_adjustment_applied": False,
                "spatial_convergence_established": False, "frequency_convergence_established": False}
    provenance = {"phase_adjustment_applied": False, "reference_planes_m": list(replay.shared.PLANES),
                  "prepared_frequency_stage": link, "solve_receipt": link,
                  **{key: stage[key] for key in replay.CONTEXT_KEYS},
                  **{name: {key: phase[key] for key in replay.EXECUTION_KEYS} for name, phase in (
                      ("frequency_equation_preparation", stage), ("linux_solve", solved), ("linux_export", exported))}}
    summary = {"checkpointed_scattering": provenance, "status": "completed_unverified", "frequency_hz": replay.FREQUENCY,
               "incident_side": "left", "hole_count": 26, "ndof": replay.DOFS, "port_dofs": 675,
               "num_incident_channels": 5, "incident_mode_indices": list(replay.INCIDENT), "power_balance_passed": True,
               "spatial_convergence_established": False, "configuration": {}, "linear_solver": report}
    return stage, solved, exported, summary, report


class H100FrequencyReplayTests(TestCase):
    def test_private_frequency_equations_preserve_all_historical_globals(self):
        before = {name: getattr(replay.shared, name) for name in ("FREQUENCY", "OMEGA", "DOFS", "INCIDENT", "RETURNING", "TRACE_SHA")}
        core = replay.frequency_core()
        self.assertEqual((core.FREQUENCY, core.DOFS, core.INCIDENT), (887500000, 307611, replay.INCIDENT))
        matrix = csc_matrix(np.array([[2 + 1j, -1j], [3j, 4]]))
        vector = np.array([[1 + 2j], [3 - 1j]])
        rhs = matrix @ vector
        core.check_original_equations(matrix, rhs, vector)
        with self.assertRaisesRegex(ValueError, "Original equation replay failed"):
            core.check_original_equations(matrix, rhs, vector * (1 + 2e-10))
        self.assertEqual(before, {name: getattr(replay.shared, name) for name in before})

    def test_archived_body_source_pairs_physics_and_distinct_map_are_strict(self):
        stage, original, config = archive_fixture()
        replay.check_body(stage, original, config, "c" * 64)
        for mutate in (lambda value: value.__setitem__("trace_sha256", replay.z4.TRACE_SHA),
                       lambda value: value.__setitem__("safe_eigensolve_performed", True),
                       lambda value: value["configuration"]["geometry"].__setitem__("thickness_nm", 201),
                       lambda value: value["array_files"]["K"].__setitem__("bytes", 11),
                       lambda value: value["implementation"]["source_sha256"].__setitem__("elastic_kernel", "b" * 64),
                       lambda value: value["driver"].__setitem__("sha256", "d" * 64)):
            wrong = deepcopy(stage)
            mutate(wrong)
            with self.assertRaises(ValueError):
                replay.check_body(wrong, original, config, "c" * 64)

    def test_accepted_chain_reuses_receipts_and_requires_retained_source_identity(self):
        # Deliberately self-contained: actual archive replay is a separate
        # recorded check, while a clean checkout must exercise these gates.
        def link(name, sha):
            return {"path": name, "sha256": sha, "bytes": 20}

        safe_link = link("safe_AP_pnc.json", replay.SAFE_SHA)
        mode_link = link("modes-887500000_AP_pnc.npz", replay.MODES_SHA)
        prebody_link = link("independent_prebody_AP_pnc.json", replay.PREBODY_SHA)
        relation_link = link("mode_relation_AP_pnc.json", replay.RELATION_SHA)
        roots = {"incident_port_roots": list(replay.INCIDENT), "backward_port_roots": list(replay.RETURNING)}
        safe = {"status": "safe_ready", "frequency_hz": replay.FREQUENCY, "trace_sha256": replay.z4.TRACE_SHA,
                "body_stage": link("stages_AP_pnc.json", replay.z4.STAGE_SHA),
                "driver": link("run_lawrence_frequency_z4_AP_pnc.py", replay.SAFE_RUNNER_SHA),
                "safe_modes": mode_link, "safe_section": link("arrays_AP_pnc.npz", replay.z4.SECTION_SHA),
                "section_receipt": link("checkpoint_AP_pnc.json", replay.z4.SECTION_RECEIPT_SHA),
                "archived_safe_modes": link("ports_AP_pnc.npz", "b" * 64),
                "implementation": {"source_sha256": {}, "packages": {}, "python": "fixture"}, **roots}
        proof = {"status": "independently_verified_safe", "frequency_hz": replay.FREQUENCY,
                 "problem_dofs": 426627, "port_dofs": 675, "all_discrete_roots_replayed": 1350,
                 "positive_open_channels": 5, "negative_open_channels": 5,
                 "safe_receipt": safe_link, "safe_modes": mode_link, "relation": relation_link,
                 "relation_status": "matched", "assigned_root_square_consistent": True, **roots,
                 **dict.fromkeys(("body_solve_performed", "body_assembly_performed", "spatial_convergence_established",
                                  "frequency_convergence_established", "continuous_branch_identity_established"), False)}
        point = {"status": "point_verified_within_saved_discretization", "frequency_hz": replay.FREQUENCY,
                 "problem_dofs": 426627, "phase_receipts": {"safe": safe_link},
                 "prebody_verification": prebody_link, "relation": relation_link}
        relation = {"status": "signed_section_relation_accepted", "frequency_hz": replay.FREQUENCY}
        stage = {**deepcopy(safe), "safe_receipt": safe_link, "safe_reference_ports": safe["archived_safe_modes"],
                 "accepted_safe_prebody": prebody_link, "safe_verification": prebody_link,
                 "signed_relation": relation_link, "accepted_reference_point": link("independent_point_AP_pnc.json", replay.POINT_SHA),
                 "safe_provider_body_stage": safe["body_stage"]}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = (root / "retained_prebody.py", root / "retained_point.py")
            for path, text in zip(sources, ("# retained prebody fixture", "# retained point fixture"), strict=True):
                path.write_text(text, encoding="utf-8")
            proof["implementation"] = {"verifier": replay.artifact(sources[0])}
            point["implementation"] = {"verifier": replay.artifact(sources[1])}
            with patch.multiple(replay, PREBODY_PRODUCER_SHA=replay.shared.file_sha(sources[0]),
                                POINT_PRODUCER_SHA=replay.shared.file_sha(sources[1])):
                accepted = replay.check_accepted_chain(stage, safe, proof, point, relation, *sources)
                self.assertFalse(accepted["safe_numerical_replay_performed"])
                for key in ("safe_modes", "accepted_safe_prebody", "signed_relation", "accepted_reference_point"):
                    wrong = deepcopy(stage)
                    wrong[key]["sha256"] = "a" * 64
                    with self.assertRaises(ValueError):
                        replay.check_accepted_chain(wrong, safe, proof, point, relation, *sources)
                wrong = deepcopy(proof)
                del wrong["all_discrete_roots_replayed"]
                with self.assertRaises(KeyError):
                    replay.check_accepted_chain(stage, safe, wrong, point, relation, *sources)
                wrong = deepcopy(point)
                wrong["implementation"]["verifier"]["sha256"] = "a" * 64
                with self.assertRaisesRegex(ValueError, "exact verifier source"):
                    replay.check_accepted_chain(stage, safe, proof, wrong, relation, *sources)
                changed = root / "changed_source.py"
                changed.write_text("# modified retained verifier", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "retained verifier source"):
                    replay.check_accepted_chain(stage, safe, proof, point, relation, changed, sources[1])

    def test_archive_schema_keeps_mesh_summary_outside_original_receipt(self):
        stage, original, config = archive_fixture()
        self.assertNotIn("mesh_checkpoint_summary", original)
        self.assertNotIn("solve_attempted", original)
        replay.check_body(stage, original, config, "c" * 64)
        wrong = deepcopy(stage)
        del wrong["mesh_summary"]
        with self.assertRaises(KeyError):
            replay.check_body(wrong, original, config, "c" * 64)
        wrong = deepcopy(stage)
        wrong["mesh_summary"]["bytes"] = 9559
        with self.assertRaisesRegex(ValueError, "exact archived body"):
            replay.check_body(wrong, original, config, "c" * 64)

    def test_new_phase_chain_rejects_changed_wrapper_gates_scope_and_executions(self):
        stage, solved, exported, summary, report = phase_fixture()
        self.assertEqual(replay.check_phases(stage, solved, exported, "c" * 64), report)
        replay.check_publication(summary, stage, solved, exported, report)
        for mutate in (lambda value: value[0]["driver"].__setitem__("sha256", "d" * 64),
                       lambda value: value[1]["linear_solver"].__setitem__("relative_target", 1e-8),
                       lambda value: value[1]["linear_solver"]["factor"].__setitem__("factor_nonzeros", 2**31),
                       lambda value: value[2].__setitem__("phase_adjustment_applied", 0),
                       lambda value: value[2].__setitem__("spatial_convergence_established", True)):
            values = deepcopy((stage, solved, exported))
            mutate(values)
            with self.assertRaises(ValueError):
                replay.check_phases(*values, "c" * 64)
        for key in ("phase_adjustment_applied", "linux_export", "prepared_frequency_stage"):
            wrong = deepcopy(summary)
            if key == "phase_adjustment_applied":
                wrong["checkpointed_scattering"][key] = None
            elif key == "linux_export":
                wrong["checkpointed_scattering"][key]["slurm_job_id"] = "other"
            else:
                wrong["checkpointed_scattering"][key]["sha256"] = "b" * 64
            with self.assertRaises(ValueError):
                replay.check_publication(wrong, stage, solved, exported, report)

    def test_actual_producer_cached_study_proxy_emits_consumable_schema(self):
        stage, solved, exported, summary, report = phase_fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "stage_AP_pnc.json"
            path.write_text(json.dumps(stage), encoding="utf-8")
            link = producer.artifact(path)
            solved["stage"] = exported["stage"] = link
            base = summary["checkpointed_scattering"]
            base["prepared_frequency_stage"] = link
            base["frequency_equation_preparation"] = {key: stage[key] for key in replay.EXECUTION_KEYS}
            base["linux_solve"] = {key: solved[key] for key in replay.EXECUTION_KEYS}
            base["linux_export"] = {key: exported[key] for key in replay.EXECUTION_KEYS}
            for key in producer.EXTRA_KEYS:
                stage.setdefault(key, {"path": "original.npz", "bytes": 2, "sha256": "b" * 64})
            path.write_text(json.dumps(stage), encoding="utf-8")
            link = producer.artifact(path)
            solved["stage"] = exported["stage"] = base["prepared_frequency_stage"] = link
            bare = {key: value for key, value in base.items() if key not in producer.EXTRA_KEYS}

            def publish(configuration, output, *, _prepared_scattering):
                return {**summary, "checkpointed_scattering": _prepared_scattering["provenance"]}

            before = producer.workflow.run_study
            with patch.object(producer.workflow, "run_study", side_effect=publish):
                core = producer.execution_core()
                actual = core.workflow.run_study({}, root, _prepared_scattering={"provenance": bare})
            self.assertIs(producer.workflow.run_study, before)
            replay.check_publication(actual, stage, solved, exported, report)
            self.assertNotIn("provenance", actual)

    def test_same_section_fields_accept_different_body_maps_and_reject_field_substitution(self):
        size = 5
        roots, backward = tuple(range(size)), tuple(range(size, 2 * size))
        mapping = {"left_dofs": np.arange(size) + 100, "right_dofs": np.arange(size) + 200}
        reference = {"left_dofs": np.arange(size) + 300, "right_dofs": np.arange(size) + 400}
        fields = {key: np.zeros(2 * size) for key in replay.z4.FIELDS}
        for item in (mapping, reference):
            item.update(deepcopy(fields))
        section = {"coordinates_yz_m": np.zeros((size, 2)), "component_dofs_xyz": np.arange(size).reshape(1, size)}
        modes = {**deepcopy(fields), "frequency_hz": np.asarray(replay.FREQUENCY),
                 "positive_indices": np.asarray(roots), "negative_indices": np.asarray(backward),
                 "propagating": np.ones(2 * size, bool), "wavenumbers_rad_m": np.arange(2 * size, dtype=np.complex128),
                 "displacements": np.zeros((size, 2 * size), np.complex128), "axial_forces": np.zeros((size, 2 * size), np.complex128)}
        digest = hashlib.sha256(json.dumps([mapping["left_dofs"].tolist(), mapping["right_dofs"].tolist()], separators=(",", ":")).encode()).hexdigest()
        with patch.multiple(replay, TRACE_SHA=digest, PORT_DOFS=size, INCIDENT=roots, RETURNING=backward):
            def load(values):
                return patch.object(replay.np, "load", side_effect=[nullcontext(value) for value in values])
            with load((section, mapping, reference, modes)):
                result, _ = replay.load_shared_ports("section", "h100", "h80", "modes")
            np.testing.assert_array_equal(result["left_dofs"], mapping["left_dofs"])
            wrong = deepcopy(reference)
            wrong["powers_w"][0] = 1
            with load((section, mapping, wrong, modes)):
                with self.assertRaisesRegex(ValueError, "archived section fields differ"):
                    replay.load_shared_ports("section", "h100", "h80", "modes")

    def test_full_p2_replay_checks_original_plane_coordinates_and_components(self):
        mesh = MeshTet.init_tensor(np.array([0., replay.shared.PLANES[1]]), np.array([-5e-7, 5e-7]), np.array([-1e-7, 1e-7]))
        basis = Basis(mesh, ElementVector(ElementTetP2()))
        components = np.empty(basis.N, int)
        for component, ids in enumerate(basis.split_indices()):
            components[ids] = component
        xyz = basis.doflocs.T
        left = np.flatnonzero(xyz[:, 0] == 0)
        right = np.array([next(i for i in np.flatnonzero(xyz[:, 0] == replay.shared.PLANES[1])
                               if components[i] == components[j] and np.array_equal(xyz[i, 1:], xyz[j, 1:])) for j in left])
        section = {"left_dofs": left, "right_dofs": right, "coordinates_yz_m": xyz[left, 1:],
                   "component_dofs_xyz": np.stack([np.flatnonzero(components[left] == i) for i in range(3)])}
        trace_sha = hashlib.sha256(json.dumps([left.tolist(), right.tolist()], separators=(",", ":")).encode()).hexdigest()
        vectors = np.zeros((basis.N, 5), complex)
        modes = {"wavenumbers_rad_m": np.array([1., -1.]), "displacements": np.zeros((len(left), 2)), "axial_forces": np.zeros((len(left), 2))}
        fields = {"points_m": mesh.p.T, "tetrahedra": mesh.t.T, "displacements_per_sqrt_w": vectors,
                  "element_order": 2, "frequencies_hz": np.full(5, replay.FREQUENCY),
                  "incident_mode_indices": replay.INCIDENT, "transmission_mode_indices": replay.INCIDENT,
                  "reflection_mode_indices": replay.RETURNING, "port_wavenumbers_rad_m": modes["wavenumbers_rad_m"],
                  "port_displacements": modes["displacements"], "port_axial_forces": modes["axial_forces"],
                  "port_left_dofs": left, "port_right_dofs": right, "dof_coordinates_m": xyz,
                  "dof_component": components, "element_dofs": basis.element_dofs, "nodal_dofs": basis.nodal_dofs}
        core = replay.frequency_core()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.npz"
            np.savez(path, points_m=mesh.p, tetrahedra=mesh.t)
            core.replay_fields(fields, path, section, modes, vectors, expected_dofs=basis.N, trace_sha256=trace_sha)
            for key in ("dof_coordinates_m", "dof_component"):
                wrong = deepcopy(fields)
                if key == "dof_coordinates_m":
                    wrong[key][left[0], 1] += 1e-12
                else:
                    wrong[key][left[0]] = (wrong[key][left[0]] + 1) % 3
                with self.assertRaises(ValueError):
                    core.replay_fields(wrong, path, section, modes, vectors, expected_dofs=basis.N, trace_sha256=trace_sha)

    def test_five_drive_physical_and_coherent_flux_checks_full_directional_projection(self):
        size = 6
        wave = np.r_[np.arange(1., 6.), 2j]
        U = np.tile(np.diag(np.sqrt(2 / (2 * np.pi * replay.FREQUENCY * abs(wave)))), (1, 2)).astype(complex)
        roots = np.r_[wave, -wave]
        F = 1j * roots * U
        incident, returning = tuple(range(5)), tuple(range(size, size + 5))
        reflection = np.diag(np.arange(1., 6.) / 10).astype(complex)
        transmission = np.diag(np.sqrt(1 - np.diag(reflection)**2) * np.exp(1j * np.arange(5) / 4))
        vectors = np.vstack((U[:, incident] + U[:, returning] @ reflection, U[:, incident] @ transmission))
        force = np.vstack((-F[:, incident] - F[:, returning] @ reflection, F[:, incident] @ transmission))
        rhs = np.vstack((-2 * F[:, incident], np.zeros((size, 5))))
        modes = {"positive_indices": np.arange(size), "negative_indices": np.arange(size, 2 * size),
                 "propagating": np.tile(np.r_[np.ones(5, bool), False], 2), "wavenumbers_rad_m": roots,
                 "displacements": U, "axial_forces": F}
        fields = {"incident_powers_w": np.ones(5)}
        for name, ids, block in (("reflection", np.arange(size, 2 * size), reflection), ("transmission", np.arange(size), transmission)):
            fields[f"{name}_all_mode_indices"] = ids
            fields[f"{name}_amplitudes"] = block
            fields[f"{name}_amplitudes_full"] = np.vstack((block, np.zeros((1, 5))))
        core = replay.frequency_core()
        core.INCIDENT, core.RETURNING = incident, returning
        values = core.replay_ports(modes, fields, np.arange(size), np.arange(size, 2 * size), vectors, force, rhs, np.linalg.norm(rhs, axis=0))
        self.assertLess(values[-1]["coherent_coefficient_power_defect_spectral_norm"], 1e-14)
        wrong = deepcopy(fields)
        wrong["reflection_amplitudes_full"][-1, 0] = .0001j
        with self.assertRaisesRegex(ValueError, "coefficients differ"):
            core.replay_ports(modes, wrong, np.arange(size), np.arange(size, 2 * size), vectors, force, rhs, np.linalg.norm(rhs, axis=0))

    def test_all_fifty_csv_joins_and_exact_frequency_are_required(self):
        r = np.eye(5, dtype=complex) * .1
        r[0, 4] = .0002 + .0003j
        t = np.eye(5, dtype=complex) * .9j
        blocks = {kind: (matrix, abs(matrix)**2) for kind, matrix in (("reflection", r), ("transmission", t))}
        R, T = (blocks[kind][1].sum(axis=0) for kind in ("reflection", "transmission"))
        totals = R, T, R, T, np.zeros(5)
        rows = [{"kind": kind, "incident_port_index": incoming, "outgoing_port_index": outgoing,
                 "incident_mode_order": col + 1, "amplitude_real": matrix[row, col].real,
                 "amplitude_imag": matrix[row, col].imag, "power_fraction": abs(matrix[row, col])**2}
                for kind, roots, matrix in (("reflection", replay.RETURNING, r), ("transmission", replay.INCIDENT, t))
                for col, incoming in enumerate(replay.INCIDENT) for row, outgoing in enumerate(roots)]
        total_rows = [{"incident_mode_order": col + 1, "frequency_GHz": .8875, "reflection": R[col], "transmission": T[col],
                       "reflection_flux": R[col], "transmission_flux": T[col], "power_balance": 1 - R[col] - T[col],
                       "flux_power_balance": 1 - R[col] - T[col]} for col in range(5)]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, values):
                with (root / name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=values[0])
                    writer.writeheader()
                    writer.writerows(values)
            write("scattering_channels_AP_pnc.csv", rows)
            write("scattering_AP_pnc.csv", total_rows)
            replay.compare_public_tables(root, blocks, totals)
            wrong = deepcopy(total_rows)
            wrong[0]["frequency_GHz"] = .1
            write("scattering_AP_pnc.csv", wrong)
            with self.assertRaisesRegex(ValueError, "frequency"):
                replay.compare_public_tables(root, blocks, totals)
            write("scattering_AP_pnc.csv", total_rows)
            wrong = deepcopy(rows)
            wrong[0] = wrong[-1]
            write("scattering_channels_AP_pnc.csv", wrong)
            with self.assertRaisesRegex(ValueError, "duplicated"):
                replay.compare_public_tables(root, blocks, totals)

    def test_full_directional_coefficient_layout_rejects_broadcasting(self):
        fields = {}
        for kind in ("reflection", "transmission"):
            fields[f"{kind}_amplitudes_full"] = np.zeros((675, 5), complex)
            fields[f"{kind}_amplitudes"] = np.zeros((5, 5), complex)
            fields[f"{kind}_all_mode_indices"] = np.arange(675)
        replay.check_coefficient_layout(fields)
        wrong = deepcopy(fields)
        wrong["reflection_amplitudes_full"] = np.zeros((1, 5), complex)
        with self.assertRaisesRegex(ValueError, "6750 directional coefficients"):
            replay.check_coefficient_layout(wrong)

    def test_wrong_external_wrapper_or_export_checksum_stops_before_body_arrays(self):
        args = SimpleNamespace(runner=Path(producer.__file__), runner_sha256="f" * 64,
                               export=Path("missing.json"), export_sha256="f" * 64)
        with patch.object(replay, "load_npz", side_effect=AssertionError("body arrays loaded")):
            with self.assertRaisesRegex(ValueError, "externally reviewed new wrapper"):
                replay.replay_point(args)
            args.runner_sha256 = replay.shared.file_sha(args.runner)
            with TemporaryDirectory() as directory:
                args.export = Path(directory) / "export.json"
                args.export.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Receipt SHA-256 differs"):
                    replay.replay_point(args)
