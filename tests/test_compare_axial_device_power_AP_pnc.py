"""Read-only native-map axial power checks. Athul Prem | _AP_pnc."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from skfem import Basis, ElementTriP2, ElementVector, MeshTri
from skfem.models.poisson import mass as scalar_mass

from tools import compare_axial_device_power_AP_pnc as comparison
from tools import map_scattering_AP_pnc as mapper


WORKFLOW = mapper.workflow


def write_native_map(directory, *, candidate=False, reflection=.04, competitor="transverse"):
    """Save analytic section polynomials and manufactured powers with native metadata."""
    output = directory / "point-001" / "attempt-fixture" / "study-fixture"
    output.mkdir(parents=True)
    mesh_path = directory / "mesh_AP_pnc.npz"
    section_path = directory / "port_section_AP_pnc.npz"
    thickness_nm = 100 if candidate else 200
    config = json.loads((mapper.ROOT / "configs/straight_scattering_AP_pnc.json").read_bytes())
    config["geometry"]["thickness_nm"] = thickness_nm
    config["analysis"]["frequency_GHz"] = 1.0
    config = WORKFLOW.resolve_config(config)
    planes = [0., config["geometry"]["length_nm"] * 1e-9]
    width, thickness = config["geometry"]["width_nm"] * 1e-9, thickness_nm * 1e-9
    ys, zs = (4, 3) if candidate else (3, 2)
    section = MeshTri.init_tensor(np.linspace(-width / 2, width / 2, ys),
                                  np.linspace(-thickness / 2, thickness / 2, zs))
    basis = Basis(section, ElementVector(ElementTriP2(), dim=3), intorder=4)
    mass = np.zeros((basis.N, basis.N))
    components = basis.split_indices()
    component_mass = scalar_mass.assemble(basis.split_bases()[0]).toarray() * config["material"]["density_kg_m3"]
    for dofs in components:
        mass[np.ix_(dofs, dofs)] = component_mass
    left, right = np.arange(basis.N), np.arange(basis.N) + basis.N
    np.savez(section_path, points_m=section.p, triangles=section.t, M=mass, left_dofs=left, right_dofs=right)
    np.savez(mesh_path, fixture=True)

    x_ids, y_ids, z_ids = components
    eta, zeta = basis.doflocs[:, x_ids] / np.array([[width], [thickness]])
    axial = np.zeros(basis.N, dtype=complex)
    axial[x_ids], axial[y_ids], axial[z_ids] = 1 + 2 * zeta + .5 * zeta**2, .25j, -.4 * zeta
    transverse = np.zeros_like(axial)
    transverse[y_ids] = eta
    if candidate:
        axial[z_ids] += .3 * eta
    if competitor == "ambiguous":
        transverse = axial + .01 * transverse
    if competitor == "no_axial":
        axial[x_ids] = 0
    incoming, profile_order = ([17, 3], [0, 1]) if candidate else ([6, 9], [1, 0])
    reflected = [1, 11]
    profiles = [axial, transverse]
    wave = np.ones(2 * basis.N, dtype=complex) * 1j
    displacement = np.zeros((basis.N, len(wave)), dtype=complex)
    forces = np.zeros_like(displacement)
    frequency_hz = 1e9
    for roots, sign in ((incoming, 1), (reflected, -1)):
        for column, (root, mode) in enumerate(zip(roots, profile_order, strict=True)):
            u = profiles[mode] * np.exp((1.17j if candidate else -.64j) * (column + 1))
            wave[root] = sign * (column + 1) * 1e6
            displacement[:, root] = u
            forces[:, root] = sign * 1j * u / (np.pi * frequency_hz * np.vdot(u, u).real)
    power_r = np.array([reflection, .2])[profile_order]
    power_t = 1 - power_r
    np.savez(output / mapper.FIELDS, port_wavenumbers_rad_m=wave, port_displacements=displacement,
             port_axial_forces=forces, port_left_dofs=left, port_right_dofs=right, element_order=2,
             incident_mode_indices=incoming, incident_powers_w=np.ones(2), reflection_mode_indices=reflected,
             transmission_mode_indices=incoming, reflection_amplitudes=1j * np.diag(np.sqrt(power_r)),
             transmission_amplitudes=np.diag(np.sqrt(power_t)).astype(complex))

    shared = {"mesh_hash": "a" * 64, "reference_planes_x_m": planes,
              "artifacts": {"mesh": mapper._link(mesh_path, directory),
                            "port_section": mapper._link(section_path, directory)}}
    inputs = {"configuration": config, "frequencies_GHz": [1.0], "hole_pattern": None}
    WORKFLOW.write_json(output / "configuration_AP_pnc.json", config)
    WORKFLOW._write_csv(output / "scattering_AP_pnc.csv", [
        {"incident_mode_order": i, "frequency_GHz": 1.0, "reflection": r, "transmission": t,
         "reflection_flux": r, "transmission_flux": t, "power_balance": 0.0,
         "flux_power_balance": 0.0, "relative_residual": 1e-13}
        for i, (r, t) in enumerate(zip(power_r, power_t, strict=True), 1)
    ])
    for filename in mapper.POINT_FILES:
        path = output / filename
        if not path.exists():
            path.write_text("Fixture\n", encoding="utf-8")
    equations = {"operator_csc_sha256": "b" * 64, "rhs_sha256": "c" * 64,
                 "relative_residual_limit": mapper.RESIDUAL_LIMIT, "backward_residual_limit": mapper.RESIDUAL_LIMIT,
                 "channels": [{"column": i, "relative_residual": 1e-13, "relative_backward_residual": 1e-15}
                              for i in range(2)]}
    summary = {"status": "completed_unverified", "run_hash": WORKFLOW.digest(config),
               "configuration": config, "analysis_kind": "scattering",
               "boundary": "matched_ports", "incident_side": "left", "frequency_hz": frequency_hz,
               "geometry_nm": config["geometry"], "hole_pattern_provenance": None, "hole_count": 0,
               "incident_mode_indices": incoming, "num_incident_channels": 2, "port_dofs": basis.N,
               "discrete_port_roots": len(wave), "maximum_relative_port_residual": 1e-13,
               "scattering_power_normalization_available": True, "power_balance_passed": True,
               "mesh_checkpoint": {"mesh_hash": shared["mesh_hash"], "mesh_sha256": mapper._sha(mesh_path)},
               "checkpointed_scattering": {"method": "frequency_map", "reference_planes_x_m": planes,
                                             "shared_sha256": WORKFLOW.digest(shared), "original_equations": equations,
                                             "port_modes_resources": {}},
               "scattering_resources": {}, "elapsed_seconds": 0.0,
               "output_sha256": {name: mapper._sha(output / name) for name in mapper.POINT_FILES}}
    summary_path = output / mapper.RUN_SUMMARY
    WORKFLOW.write_json(summary_path, summary)
    point = {"index": 1, "frequency_GHz": 1.0, "status": "accepted",
             "attempts": [{"path": output.parent.relative_to(directory).as_posix(), "status": "accepted"}],
             "accepted": {"summary": mapper._link(summary_path, directory)}}
    state = {**mapper.AUTHOR, **mapper.CONVERGENCE, "schema_version": 1, "inputs": inputs, "source_git_commit": "0" * 40,
             "inputs_sha256": WORKFLOW.digest(inputs), "shared": shared, "points": [point]}
    mapper._write_checkpoint(directory / mapper.CHECKPOINT, state)
    mapper._publish(state, None, {1: mapper._read_accepted(point, config, shared, directory)}, directory)


class AxialDevicePowerTests(TestCase):
    def setUp(self):
        # Windows runners may expose TEMP through a short-name alias.
        directory = Path(self.enterContext(TemporaryDirectory())).resolve()
        self.reference = directory / "reference"
        self.candidate = directory / "candidate"

    def test_cli_matches_native_roots_on_unequal_normalized_sections_and_reports_power_change(self):
        write_native_map(self.reference)
        write_native_map(self.candidate, candidate=True, reflection=.06)
        before = {path: path.read_bytes() for path in self.reference.parent.rglob("*") if path.is_file()}
        with redirect_stdout(StringIO()) as stdout:
            self.assertEqual(comparison.main([str(self.reference), str(self.candidate), "--frequency-ghz", "1"]), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "matched")
        self.assertEqual((result["reference"]["drive"], result["reference"]["root"]), (2, 9))
        self.assertEqual((result["candidate"]["drive"], result["candidate"]["root"]), (1, 17))
        # Analytic mean energies on [-1/2, 1/2] in each normalized coordinate.
        x_energy = 1 + 5 / 12 + .25 / 80
        total_energy = x_energy + .25**2 + .4**2 / 12
        candidate_energy = total_energy + .3**2 / 12
        self.assertAlmostEqual(result["reference"]["axial_kinetic_energy_fraction"], x_energy / total_energy)
        self.assertAlmostEqual(result["candidate"]["axial_kinetic_energy_fraction"], x_energy / candidate_energy)
        for name in ("reference_to_candidate_overlap", "candidate_to_reference_overlap"):
            self.assertAlmostEqual(result["match"][name], np.sqrt(total_energy / candidate_energy), places=12)
        for label, r in (("reference", .04), ("candidate", .06)):
            self.assertAlmostEqual(result[label]["R"], r)
            self.assertAlmostEqual(result[label]["T"], 1 - r)
        self.assertAlmostEqual(result["delta"]["R"]["percentage_points"], 2)
        self.assertAlmostEqual(result["delta"]["R"]["relative_change"], .5)
        self.assertAlmostEqual(result["delta"]["T"]["percentage_points"], -2)
        self.assertAlmostEqual(result["delta"]["T"]["relative_change"], -.02 / .96)
        after = {path: path.read_bytes() for path in self.reference.parent.rglob("*") if path.is_file()}
        self.assertEqual(after, before)

    def test_keeps_ambiguity_or_absence_of_axial_predominance_unresolved(self):
        write_native_map(self.reference)
        for condition in ("ambiguous", "no_axial"):
            with self.subTest(condition=condition):
                candidate = self.candidate / condition
                write_native_map(candidate, candidate=True, competitor=condition)
                self.assertEqual(comparison.compare_axial_power(self.reference, candidate, 1)["status"], "unresolved")

    def test_zero_reference_reflection_keeps_absolute_change_and_leaves_relative_change_undefined(self):
        write_native_map(self.reference, reflection=0.)
        write_native_map(self.candidate, candidate=True, reflection=.06)
        result = comparison.compare_axial_power(self.reference, self.candidate, 1)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["reference"]["R"], 0.)
        self.assertAlmostEqual(result["delta"]["R"]["percentage_points"], 6)
        self.assertIsNone(result["delta"]["R"]["relative_change"])
        self.assertEqual(result["delta"]["R"]["relative_change_status"], "zero_reference")

    def test_modified_shared_section_is_rejected(self):
        write_native_map(self.reference)
        write_native_map(self.candidate, candidate=True)
        path = self.candidate / "port_section_AP_pnc.npz"
        with path.open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "port_section_AP_pnc.npz"):
            comparison.compare_axial_power(self.reference, self.candidate, 1)
