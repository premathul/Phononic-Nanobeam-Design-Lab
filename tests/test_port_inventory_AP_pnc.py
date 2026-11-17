"""Closed-form SAFE inventory checks. Author: Athul Prem | _AP_pnc."""

from dataclasses import replace
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
from skfem import Basis, ElementTriP2, ElementVector, MeshTri

from tools import run_lawrence_port_inventory_AP_pnc as inventory
from phononic.port_modes_AP_pnc import PortModes, PortSystem, solve_port_modes


LENGTH_M = 1e-6
ANCHOR_HZ = 1_000_000_000
OMEGA_0 = 2 * np.pi * ANCHOR_HZ
REFERENCE_SPEED_M_S = 5000.0


def closed_form_section():
    mesh = MeshTri(np.array([[0, LENGTH_M, 0], [0, 0, 0.2 * LENGTH_M]]),
                   np.array([[0], [1], [2]]))
    basis = Basis(mesh, ElementVector(ElementTriP2(), dim=3))
    n = basis.N
    # Two helicity branches and N-2 independent scalar branches. The nonzero
    # symmetric part of G contributes to traction even though it cancels from
    # the quadratic eigenproblem. Scalar trace 3 has a true 2.8 GHz cutoff.
    transverse = np.r_[25, 25, 0, 2.8**2, np.arange(43, 43 + n - 4)]
    axial = np.r_[1., 1., np.arange(1., n - 1)]
    coupling = np.zeros((n, n))
    coupling[:2, :2] = [[1, 6], [1, -2]]
    port = PortSystem(None, basis, np.arange(n), np.arange(n, 2 * n),
                      np.diag(transverse), np.eye(n) / OMEGA_0**2,
                      np.diag(axial) * LENGTH_M**2, coupling * LENGTH_M,
                      REFERENCE_SPEED_M_S)
    return port, transverse, axial


def closed_form_modes(section, frequency_hz):
    """Use independent scalar quadratics and the dispersion slope for power."""
    port, transverse, axial = section
    n = port.basis.N
    frequency_ratio = frequency_hz / ANCHOR_HZ
    canonical_vectors = np.eye(n, dtype=complex)
    canonical_vectors[:2, :2] = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
    slopes = np.r_[5, -5, np.zeros(n - 2)]
    dimensionless_k = np.concatenate([
        np.roots([q, h, a - frequency_ratio**2])
        for a, h, q in zip(transverse, slopes, axial)
    ]).astype(complex)
    k = dimensionless_k / LENGTH_M
    displacements = np.repeat(canonical_vectors, 2, axis=1)
    # Energy per unit length times the analytic group velocity.
    signed_powers = (OMEGA_0 * LENGTH_M * frequency_ratio / 4
                     * (np.repeat(slopes, 2) + 2 * np.repeat(axial, 2) * dimensionless_k.real))
    propagating = (k.imag == 0)
    directions = np.where(propagating, np.sign(signed_powers), np.sign(k.imag))
    reference_power = REFERENCE_SPEED_M_S * frequency_ratio**2 / 2
    displacements /= np.sqrt(np.where(propagating, abs(signed_powers), reference_power))
    forces = displacements * (1j * k * np.repeat(axial, 2) * LENGTH_M**2)
    forces[:2] += LENGTH_M * np.vstack((displacements[0] + 6 * displacements[1],
                                        displacements[0] - 2 * displacements[1]))

    def ordered_indices(direction):
        indices = np.flatnonzero(directions == direction)
        return indices[np.lexsort((abs(k[indices]), ~propagating[indices]))]

    return PortModes(port, frequency_hz, k, displacements, forces, ordered_indices(1),
                     ordered_indices(-1), propagating, np.where(propagating, directions, 0.),
                     np.zeros(2 * n))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(path):
    return path.read_bytes(), path.stat().st_mtime_ns


class PortInventoryTests(TestCase):
    def setUp(self):
        self.section = closed_form_section()
        self.port = self.section[0]
        self.anchor = closed_form_modes(self.section, ANCHOR_HZ)
        self.provenance = {"archive": {"ports": {"path": "analytic-fixture.npz"}},
                           "source": {"fixture": "initial"}}

    def assert_analytic_point(self, path, frequency_hz, count):
        expected = closed_form_modes(self.section, frequency_hz)
        n = self.port.basis.N
        with np.load(path, allow_pickle=False) as actual:
            k = actual["wavenumbers_rad_m"]
            matches, reference = linear_sum_assignment(abs(k[:, None] - expected.wavenumbers_rad_m))
            self.assertEqual(len(matches), 2 * n)
            np.testing.assert_allclose(k[matches], expected.wavenumbers_rad_m[reference], rtol=5e-12)
            displacements, forces = actual["displacements"], actual["axial_forces"]
            velocity = -2j * np.pi * frequency_hz * displacements
            for name, sign in (("positive_indices", 1), ("negative_indices", -1)):
                direction = actual[name]
                propagating = actual["propagating"][direction]
                self.assertEqual(direction.size, n)
                self.assertEqual(np.count_nonzero(propagating), count)
                self.assertTrue(np.all(sign * k[direction[~propagating]].imag > 0))
                work = velocity[:, direction].conj().T @ forces[:, direction]
                flux = -0.25 * (work + work.conj().T)
                np.testing.assert_allclose(flux, np.diag(sign * propagating.astype(float)), atol=1e-10)
            # Distinct analytic polarizations fix the shape up to complex gauge.
            # Compare traction and displacement after using that same gauge.
            analytic_u = expected.displacements[:, reference]
            root_gauge = np.sum(analytic_u.conj() * displacements[:, matches], axis=0)
            root_gauge /= np.sum(abs(analytic_u)**2, axis=0)
            np.testing.assert_allclose(abs(root_gauge), 1, atol=1e-11)
            np.testing.assert_allclose(displacements[:, matches], analytic_u * root_gauge, atol=1e-13)
            np.testing.assert_allclose(forces[:, matches], expected.axial_forces[:, reference] * root_gauge,
                                       atol=1e-17, rtol=1e-11)
            self.assertLessEqual(actual["relative_residuals"].max(), 1e-8)

    def test_cutoff_failure_and_interrupted_point_preserve_accepted_results(self):
        def interrupt_before_last_sample(port, frequency):
            if frequency == 5_200_000_000:
                raise KeyboardInterrupt("test interruption")
            return solve_port_modes(port, frequency)

        with TemporaryDirectory() as directory:
            output = Path(directory) / "inventory"
            run = partial(inventory.run_inventory, output, self.port, self.anchor, self.provenance)
            with patch.object(inventory, "solve_port_modes", side_effect=interrupt_before_last_sample) as solver:
                with self.assertRaisesRegex(KeyboardInterrupt, "test interruption"):
                    run()
            called = [call.args[1] for call in solver.call_args_list]
            self.assertEqual(called, [100_000_000, 2_800_000_000, 5_200_000_000])
            low_modes = output / "modes-100000000_AP_pnc.npz"
            accepted = (low_modes, output / "point-100000000_AP_pnc.json")
            accepted_snapshot = [snapshot(path) for path in accepted]
            summary_path = output / "port_inventory_AP_pnc.json"
            first = read_json(summary_path)
            self.assertEqual(first["status"], "incomplete")
            self.assertFalse(first["spatially_validated"])
            self.assertFalse(first["frequency_validated"])
            self.assertEqual([row["status"] for row in first["points"]],
                             ["accepted", "archived", "failed", "interrupted", "pending"])
            self.assertEqual(first["points"][2]["cutoff_status"], "unresolved")
            self.assertFalse(first["points"][1]["eigensolve_performed"])
            self.assertFalse((output / "modes-1000000000_AP_pnc.npz").exists())
            for frequency in (2_800_000_000, 5_200_000_000):
                self.assertFalse((output / f"modes-{frequency}_AP_pnc.npz").exists())

            with patch.object(inventory, "solve_port_modes", wraps=solve_port_modes) as solver:
                run(resume=True)
                self.assertEqual([call.args[1] for call in solver.call_args_list], [6_000_000_000])
            self.assertEqual([snapshot(path) for path in accepted], accepted_snapshot)
            after = read_json(summary_path)
            self.assertEqual([row["status"] for row in after["points"]],
                             ["accepted", "archived", "failed", "interrupted", "accepted"])
            self.assertEqual(after["points"][:4], first["points"][:4])
            self.assert_analytic_point(low_modes, 100_000_000, 1)
            self.assert_analytic_point(output / "modes-6000000000_AP_pnc.npz", 6_000_000_000, 4)

            publication = [path for path in output.iterdir() if path.is_file()]
            publication_snapshot = [snapshot(path) for path in publication]
            with patch.object(inventory, "solve_port_modes", side_effect=AssertionError("accepted points were rerun")):
                run(resume=True)
            self.assertEqual([snapshot(path) for path in publication], publication_snapshot)

    def test_resume_refuses_changed_inputs_source_and_result(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "inventory"
            run = partial(inventory.run_inventory, output, self.port, self.anchor)
            run(self.provenance)
            summary_path = output / "port_inventory_AP_pnc.json"
            summary_snapshot = snapshot(summary_path)
            # A symmetric addition to G changes traction without changing the
            # quadratic polynomial or modal power, so both anchors are valid.
            delta_g = np.eye(self.port.basis.N) * LENGTH_M
            changed_section = replace(self.port, G=self.port.G + delta_g)
            changed_anchor = replace(self.anchor, port=changed_section,
                                     axial_forces=self.anchor.axial_forces + delta_g @ self.anchor.displacements)
            inventory.validate_modes(changed_anchor)
            with patch.object(inventory, "solve_port_modes", side_effect=AssertionError("refusal must precede eigensolve")):
                with self.assertRaisesRegex(ValueError, r"(?i)identity|inputs|implementation"):
                    run(dict(self.provenance, source={"fixture": "changed"}), resume=True)
                with self.assertRaisesRegex(ValueError, r"(?i)identity|inputs|implementation"):
                    inventory.run_inventory(output, changed_section, changed_anchor, self.provenance, resume=True)
                point_path = output / "point-100000000_AP_pnc.json"
                point_bytes = point_path.read_bytes()
                altered_point = read_json(point_path)
                altered_point["positive_open_channels"] += 1
                point_path.write_text(json.dumps(altered_point), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, r"(?i)receipt|point"):
                    run(self.provenance, resume=True)
                point_path.write_bytes(point_bytes)
                modes = output / "modes-100000000_AP_pnc.npz"
                with modes.open("ab") as stream:
                    stream.write(b"changed result")
                with self.assertRaises((RuntimeError, ValueError)):
                    run(self.provenance, resume=True)
            self.assertEqual(snapshot(summary_path), summary_snapshot)

    def test_archive_rejects_changed_receipt_section_and_source_before_array_load(self):
        implementation = {"source_sha256": {"fixture": "original"}, "packages": {}, "python": "fixture"}
        ports_bytes = b"opaque SAFE NPZ for provenance checks"
        ports_sha256 = sha256(ports_bytes).hexdigest()
        receipt = {"status": "safe_ready", "trace_dofs": 975, "frequency_hz": ANCHOR_HZ,
                   "discrete_roots": 1950, "ports": {"sha256": ports_sha256, "bytes": len(ports_bytes)},
                   "implementation": implementation}
        receipt_bytes = json.dumps(receipt).encode("utf-8")
        with TemporaryDirectory() as directory:
            archive = Path(directory)
            receipt_path = archive / inventory.SAFE
            port_path = archive / inventory.PORTS
            with (patch.object(inventory, "SAFE_SHA256", sha256(receipt_bytes).hexdigest()),
                  patch.object(inventory, "PORTS_SHA256", ports_sha256),
                  patch.object(np, "load", side_effect=AssertionError("unqualified input reached array load"))):
                port_path.write_bytes(ports_bytes)
                receipt_path.write_bytes(receipt_bytes + b" ")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    inventory.read_archive(archive, implementation)
                receipt_path.write_bytes(receipt_bytes)
                port_path.write_bytes(ports_bytes + b" ")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    inventory.read_archive(archive, implementation)
                port_path.write_bytes(ports_bytes)
                with self.assertRaisesRegex(ValueError, "source"):
                    inventory.read_archive(archive, dict(implementation, source_sha256={"fixture": "changed"}))

    def test_validation_rejects_wrong_original_equations_and_signed_basis(self):
        inventory.validate_modes(self.anchor)
        positive_root = self.anchor.positive_indices[0]
        negative_root = self.anchor.negative_indices[0]
        # Change an eigenvector while maintaining traction consistency. The
        # independent equilibrium gate must still catch the wrong eigenpair.
        incorrect = self.anchor.displacements.copy()
        incorrect[0, positive_root] += 0.1
        k = self.anchor.wavenumbers_rad_m
        consistent_traction = self.port.G @ incorrect + (self.port.Q @ incorrect) * (1j * k)
        with self.assertRaisesRegex((RuntimeError, ValueError), r"(?i)residual|equation"):
            inventory.validate_modes(replace(self.anchor, displacements=incorrect, axial_forces=consistent_traction))

        # Flipping a pair's direction leaves its polynomial and physical
        # traction unchanged. Signed energy flow must reject that labeling.
        positive = self.anchor.positive_indices.copy()
        negative = self.anchor.negative_indices.copy()
        positive[0], negative[0] = negative_root, positive_root
        with self.assertRaisesRegex((RuntimeError, ValueError), r"(?i)direction|power|flux"):
            inventory.validate_modes(replace(self.anchor, positive_indices=positive, negative_indices=negative))
