"""Independent bounded-frequency replay fixtures. Athul Prem | _AP_pnc."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from scipy.sparse import csc_matrix

from tools import replay_lawrence_device_point_AP_pnc as shared
from tools import replay_lawrence_frequency_point_AP_pnc as replay


def stage_fixture():
    def link(name):
        return {"path": f"/archive/{name}", "bytes": 10, "sha256": name * 2}
    original_identity = {"source_sha256": {"kernel": "same"}, "packages": {"numpy": "same"},
                         "python": "3.12.12", "platform": "Linux-archived"}
    modes_identity = {**original_identity, "python": "3.12.14", "platform": "Windows-producer"}
    original = {"mesh": link("mesh.npz"), "mesh_summary": link("summary.json"),
                "array_files": {key: link(f"{key}.npz") for key in ("K", "M")},
                "safe_modes": link("ports.npz"), "safe_receipt": link("safe.json"),
                "configuration": {"geometry": {"thickness": 200}, "analysis": {"frequency_GHz": 1}, "resources": {}},
                "implementation": original_identity}
    inputs = {"identity": {"archive": {"section_array_sha256": "section", "original_implementation": original_identity,
                                       "current_implementation": modes_identity}}}
    point = {"modes": link("modes.npz"), "metrics": {"open_channels": [{"root_id": root} for root in replay.INCIDENT]}}
    stage = {**deepcopy(original), "status": "operator_staged", "frequency_hz": replay.FREQUENCY,
             "displacement_dofs": replay.DOFS, "trace_sha256": replay.TRACE_SHA,
             "reference_planes_m": list(replay.PLANES), "body_stage": {"sha256": replay.BODY_STAGE_SHA},
             "safe_section": original["safe_modes"], "safe_modes": point["modes"],
             "safe_section_array_sha256": "section", "interval_inputs": {"sha256": replay.INPUTS_SHA},
             "interval_point": {"sha256": replay.POINT_SHA}, "interval_ledger": {"sha256": replay.LEDGER_SHA},
             "incident_port_roots": list(replay.INCIDENT), "backward_port_roots": list(replay.RETURNING),
             "open_channels": point["metrics"]["open_channels"], "cutoff_status": "solver_resolved_at_sample",
             "degeneracy_status": "none_detected", "original_section_implementation": original_identity,
             "section_mode_implementation": modes_identity,
             "implementation": {**original_identity, "platform": "Linux-current"}}
    stage["configuration"]["analysis"]["frequency_GHz"] = replay.FREQUENCY / 1e9
    stage.update(dict.fromkeys(("mesh_generated", "body_assembly_performed", "safe_eigensolve_performed",
                               "factorization_performed", "phase_adjustment_applied", "spatial_convergence_established",
                               "frequency_convergence_established"), False))
    return stage, original, inputs, point


def relation_fixture():
    roots, reverse = np.arange(5), np.arange(5, 10)
    identity = np.eye(5, dtype=complex)
    mass = np.diag(np.arange(1, 6, dtype=float))
    order = np.array([3, 0, 4, 2, 1])
    gauge = np.exp(1j * np.arange(5))
    first = {"incoming": {"roots": roots, "k": np.arange(1, 6.), "displacements": identity},
             "reflection": {"roots": reverse, "k": -np.arange(1, 6.), "displacements": identity},
             "reverse_partners": dict(zip(roots, reverse, strict=True))}
    second = {name: {"roots": first[name]["roots"] + 100, "k": first[name]["k"][order] * 1.1,
                     "displacements": identity[:, order] * gauge} for name in ("incoming", "reflection")}
    second["reverse_partners"] = {int(root + 100): int(reverse[index] + 100) for index, root in enumerate(roots)}
    # Construct the expected result analytically, without the producer's matcher.
    cross = identity[:, order] * gauge
    aligned = np.argsort(order)
    groups = {}
    for name in ("incoming", "reflection"):
        groups[name] = {"status": "matched", "reference_roots": first[name]["roots"].tolist(),
                        "candidate_roots_unordered": second[name]["roots"].tolist(),
                        "full_normalized_complex_overlap": np.stack((cross.real, cross.imag), axis=-1).tolist(),
                        "full_mass_overlap_matrix": abs(cross).tolist(), "candidate_order": aligned.tolist(),
                        "candidate_roots": second[name]["roots"][aligned].tolist(), "mass_overlaps": [1.] * 5,
                        "candidate_gauge_phase_rad": np.angle(gauge[aligned]).tolist(),
                        "candidate_gauge_complex": np.stack((gauge[aligned].real, gauge[aligned].imag), axis=-1).tolist()}
    groups["transmission"] = deepcopy(groups["incoming"])
    return first, second, mass, {"status": "matched", "matched_channels": groups,
        "reverse_partner_consistency": {"status": "consistent", "reference_forward_to_reverse": first["reverse_partners"],
                                        "candidate_forward_to_reverse": second["reverse_partners"]}}


class FrequencyPointReplayTests(TestCase):
    def test_private_frequency_gates_preserve_archived_defaults(self):
        core = replay.frequency_core()
        self.assertEqual(core.FREQUENCY, 887_500_000)
        self.assertEqual(core.OMEGA, 2 * np.pi * 887_500_000)
        self.assertEqual(core.INCIDENT, replay.INCIDENT)
        self.assertEqual(shared.FREQUENCY, 100_000_000)
        self.assertEqual(shared.INCIDENT, (1948, 1916, 1940, 1873))
        operator = csc_matrix(np.array([[2 + 1j, -1j], [3j, 4]]))
        vector = np.array([[1 + 2j], [3 - 1j]])
        rhs = operator @ vector
        core.check_original_equations(operator, rhs, vector)
        with self.assertRaisesRegex(ValueError, "Original equation replay failed"):
            core.check_original_equations(operator, rhs, vector * (1 + 2e-10))

    def test_stage_keeps_exact_frequency_body_channels_and_physics(self):
        args = stage_fixture()
        replay.check_stage_identity(*args)
        for key, value in (("frequency_hz", 1_000_000_000), ("incident_port_roots", list(reversed(replay.INCIDENT))),
                           ("phase_adjustment_applied", True)):
            changed = deepcopy(args)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay.check_stage_identity(*changed)
        changed = deepcopy(args)
        changed[0]["array_files"]["K"]["sha256"] = "other"
        with self.assertRaises(ValueError):
            replay.check_stage_identity(*changed)
        changed = deepcopy(args)
        changed[0]["configuration"]["geometry"]["thickness"] = 201
        with self.assertRaisesRegex(ValueError, "physical inputs"):
            replay.check_stage_identity(*changed)

    def test_separate_windows_mode_and_linux_section_provenance_is_required(self):
        args = stage_fixture()
        args[0]["section_mode_implementation"] = deepcopy(args[0]["original_section_implementation"])
        with self.assertRaisesRegex(ValueError, "conflated"):
            replay.check_stage_identity(*args)
        args = stage_fixture()
        args[0]["implementation"]["source_sha256"] = {"kernel": "changed"}
        with self.assertRaisesRegex(ValueError, "provenance"):
            replay.check_stage_identity(*args)

    def test_five_channel_projection_and_coherent_nodal_flux_at_new_frequency(self):
        core = replay.frequency_core()
        identity = np.eye(5, dtype=complex)
        force_unit = identity * (1j / (np.pi * replay.FREQUENCY))
        positive, negative = np.arange(5), np.arange(5, 10)
        modes = {"positive_indices": positive, "negative_indices": negative, "propagating": np.ones(10, dtype=bool),
                 "wavenumbers_rad_m": np.r_[np.arange(1, 6), -np.arange(1, 6)],
                 "displacements": np.tile(identity, (1, 2)), "axial_forces": np.column_stack((force_unit, -force_unit))}
        fields = {"incident_powers_w": np.ones(5)}
        for name, roots, amplitude in (("reflection", negative, .6 * identity), ("transmission", positive, .8j * identity)):
            fields[f"{name}_all_mode_indices"] = roots
            fields[f"{name}_amplitudes"] = fields[f"{name}_amplitudes_full"] = amplitude.copy()
        vectors = np.vstack((1.6 * identity, np.zeros_like(identity), .8j * identity))
        forces = np.vstack((-.4 * force_unit, np.zeros_like(identity), .8j * force_unit))
        rhs = np.vstack((-2 * force_unit, np.zeros((10, 5), dtype=complex)))
        arguments = (modes, fields, np.arange(5), np.arange(10, 15), vectors, forces, rhs, np.linalg.norm(rhs, axis=0))
        result = core.replay_ports(*arguments, incident_roots=positive, returning_roots=negative)
        self.assertLess(result[5]["coherent_coefficient_power_defect_spectral_norm"], 1e-14)
        np.testing.assert_allclose(result[1][0], .36, atol=1e-15)
        np.testing.assert_allclose(result[1][1], .64, atol=1e-15)
        fields["reflection_amplitudes_full"][0, 1] = .01j
        with self.assertRaisesRegex(ValueError, "coefficients differ"):
            core.replay_ports(*arguments, incident_roots=positive, returning_roots=negative)
        fields["reflection_amplitudes_full"][0, 1] = 0
        forces[5, 0] = .01
        with self.assertRaisesRegex(ValueError, "boundary forces fail"):
            core.replay_ports(*arguments, incident_roots=positive, returning_roots=negative)

    def test_per_column_conservation_cannot_mask_coherent_defect(self):
        core = replay.frequency_core()
        identity = np.eye(2)
        force = identity * (1j / (np.pi * replay.FREQUENCY))
        transmission = np.array([[1., 1.], [0., 0.]])
        zeros, ones = np.zeros(2), np.ones(2)
        core.check_power(ones, zeros, ones, zeros, ones, zeros)
        with self.assertRaisesRegex(ValueError, "coherent power defect"):
            core.check_coherent_power(np.zeros((2, 2)), transmission, (identity, -force), (identity, force),
                                      (identity, -force), (transmission, force @ transmission),
                                      incident_roots=(0, 1), returning_roots=(2, 3))

    def test_independent_exhaustive_relation_checks_gauges_and_stop_decision(self):
        first, second, mass, published = relation_fixture()
        result = replay.check_relation_arrays(published, first, second, mass)
        self.assertEqual(result["relation_status"], "matched")
        self.assertLess(result["overlap_checks"]["incoming"]["maximum_complex_overlap_difference"], 1e-15)
        broken = deepcopy(published)
        broken["matched_channels"]["incoming"]["candidate_gauge_complex"][0][0] += .01
        with self.assertRaisesRegex(ValueError, "complex gauge"):
            replay.check_relation_arrays(broken, first, second, mass)
        broken = deepcopy(published)
        broken["matched_channels"]["incoming"]["candidate_gauge_phase_rad"][0] += .01
        with self.assertRaisesRegex(ValueError, "complex gauge"):
            replay.check_relation_arrays(broken, first, second, mass)
        broken = deepcopy(published)
        broken["reverse_partner_consistency"]["reference_forward_to_reverse"][0] = 9
        with self.assertRaisesRegex(ValueError, "reverse-partner map"):
            replay.check_relation_arrays(broken, first, second, mass)
        second["incoming"]["displacements"][:, 0] = first["incoming"]["displacements"][:, :2].sum(axis=1)
        measured = replay.independent_relation_group(first["incoming"], second["incoming"], mass)
        self.assertEqual(measured["status"], "unresolved")
        with self.assertRaisesRegex(ValueError, "overlap values"):
            replay.check_relation_arrays(published, first, second, mass)

    def test_public_tables_require_all_fifty_complex_entries_and_exact_frequency(self):
        reflection, transmission = .6 * np.eye(5), .8j * np.eye(5)
        blocks = {"reflection": (reflection, abs(reflection)**2), "transmission": (transmission, abs(transmission)**2)}
        totals = (np.full(5, .36), np.full(5, .64), np.full(5, .36), np.full(5, .64), np.zeros(5))
        channels = [{"kind": kind, "incident_port_index": incoming, "outgoing_port_index": outgoing,
                     "incident_mode_order": column + 1, "amplitude_real": block[0][row, column].real,
                     "amplitude_imag": block[0][row, column].imag, "power_fraction": block[1][row, column]}
                    for kind, roots in (("reflection", replay.RETURNING), ("transmission", replay.INCIDENT))
                    for column, incoming in enumerate(replay.INCIDENT) for row, outgoing in enumerate(roots)
                    for block in (blocks[kind],)]
        powers = [{"incident_mode_order": column + 1, "frequency_GHz": .8875,
                   "reflection": .36, "transmission": .64, "reflection_flux": .36, "transmission_flux": .64,
                   "power_balance": 0., "flux_power_balance": 0.} for column in range(5)]
        with TemporaryDirectory() as directory:
            public = Path(directory)
            def write(name, rows):
                with (public / name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            write("scattering_channels_AP_pnc.csv", channels)
            write("scattering_AP_pnc.csv", powers)
            replay.compare_public_tables(public, blocks, totals)
            powers[0]["frequency_GHz"] = .1
            write("scattering_AP_pnc.csv", powers)
            with self.assertRaisesRegex(ValueError, "frequency"):
                replay.compare_public_tables(public, blocks, totals)
            powers[0]["frequency_GHz"] = .8875
            write("scattering_AP_pnc.csv", powers)
            channels[1] = channels[0]
            write("scattering_channels_AP_pnc.csv", channels)
            with self.assertRaisesRegex(ValueError, "duplicated"):
                replay.compare_public_tables(public, blocks, totals)

    def test_full_p2_frequency_vector_and_element_identity(self):
        core = replay.frequency_core()
        length = replay.PLANES[1]
        points = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                           [length, 0., 0.], [length, 1., 0.], [length, 0., 1.]])
        tetrahedra = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])
        sites = [(index, index) for index in range(6)]
        tetra_sites = []
        order = ((0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
        for tetra in tetrahedra:
            ids = []
            for a, b in order:
                edge = tuple(sorted((int(tetra[a]), int(tetra[b]))))
                if edge not in sites:
                    sites.append(edge)
                ids.extend(range(3 * sites.index(edge), 3 * sites.index(edge) + 3))
            tetra_sites.append(ids)
        locations = np.array([(points[a] + points[b]) / 2 for a, b in sites])
        coordinates = np.repeat(locations, 3, axis=0)
        components = np.tile(np.arange(3), len(sites))
        left = np.flatnonzero(coordinates[:, 0] == 0)
        right_ids = np.flatnonzero(coordinates[:, 0] == length)
        right = np.array([next(index for index in right_ids if components[index] == components[source]
                               and np.array_equal(coordinates[index, 1:], coordinates[source, 1:])) for source in left])
        trace = hashlib.sha256(json.dumps([left.tolist(), right.tolist()], separators=(",", ":")).encode()).hexdigest()
        section = {"left_dofs": left, "right_dofs": right, "coordinates_yz_m": coordinates[left, 1:],
                   "component_dofs_xyz": np.array([np.flatnonzero(components[left] == value) for value in range(3)])}
        modes = {"wavenumbers_rad_m": np.array([1, -1]), "displacements": np.ones((len(left), 2)),
                 "axial_forces": np.ones((len(left), 2), dtype=complex)}
        vectors = np.arange(len(coordinates), dtype=complex)[:, None]
        fields = {"points_m": points, "tetrahedra": tetrahedra, "displacements_per_sqrt_w": vectors,
                  "element_order": np.array(2), "frequencies_hz": np.array([replay.FREQUENCY]),
                  "incident_mode_indices": np.array([0]), "transmission_mode_indices": np.array([0]),
                  "reflection_mode_indices": np.array([1]), "port_wavenumbers_rad_m": modes["wavenumbers_rad_m"],
                  "port_displacements": modes["displacements"], "port_axial_forces": modes["axial_forces"],
                  "port_left_dofs": left, "port_right_dofs": right, "dof_coordinates_m": coordinates,
                  "dof_component": components, "element_dofs": np.array(tetra_sites).T,
                  "nodal_dofs": np.arange(18).reshape(6, 3).T}
        with TemporaryDirectory() as directory:
            mesh = Path(directory) / "mesh.npz"
            np.savez(mesh, points_m=points.T, tetrahedra=tetrahedra.T)
            def check():
                core.replay_fields(fields, mesh, section, modes, vectors, expected_dofs=len(coordinates),
                                   incident_roots=(0,), returning_roots=(1,), trace_sha256=trace)
            check()
            fields["frequencies_hz"][0] = 100_000_000
            with self.assertRaisesRegex(ValueError, "frequencies_hz"):
                check()
            fields["frequencies_hz"][0] = replay.FREQUENCY
            fields["dof_component"][0] = 2
            with self.assertRaisesRegex(ValueError, "Cartesian field components"):
                check()
