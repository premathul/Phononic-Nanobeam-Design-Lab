"""Fixed-frequency Bloch factor benchmarks. Author: Athul Prem | _AP_pnc."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm as sparse_norm
from skfem import Basis, ElementTetP1, ElementVector, MeshTet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from phononic.elasticity_AP_pnc import ElasticSystem, assemble_elasticity
from phononic.lead_modes_AP_pnc import solve_bloch
from phononic.material_AP_pnc import CubicMaterial
from bloch_factors_AP_pnc import solve_bloch_factors


PERIOD_M = 600e-9
SPRING_N_M = 2.0
SITE_MASS_KG = 2e-18
FREQUENCY_SCALE_HZ = math.sqrt(SPRING_N_M / SITE_MASS_KG) / math.tau


def _spring_cell():
    """Embed uncoupled two-spring chains in matched x-face vector DOF order."""
    x = [0.0, PERIOD_M / 2, PERIOD_M]
    mesh = MeshTet.init_tensor(x, [-250e-9, 250e-9], [-100e-9, 100e-9])
    basis = Basis(mesh, ElementVector(ElementTetP1()))
    locations = basis.doflocs
    chains = []
    for component in basis.split_indices():
        planes = []
        for coordinate in x:
            dofs = component[locations[0, component] == coordinate]
            planes.append(dofs[np.lexsort(locations[1:, dofs])])
        chains.extend(np.column_stack(planes))

    stiffness = np.zeros((basis.N, basis.N))
    masses = np.empty(basis.N)
    # Sharing the endpoints between cells gives each physical site its full mass.
    weights = np.array([0.5, 1.0, 0.5])
    spring = SPRING_N_M * np.array([[1.0, -1.0], [-1.0, 1.0]])
    for channel, sites in enumerate(chains):
        masses[sites] = SITE_MASS_KG * weights
        # Pin every other chain above the test band so only one branch propagates.
        stiffness[sites, sites] += 8 * channel * SPRING_N_M * weights
        for pair in (sites[:2], sites[1:]):
            indices = np.ix_(pair, pair)
            stiffness[indices] += spring
    system = ElasticSystem(mesh, basis, csr_matrix(stiffness), diags(masses, format="csr"))
    return system, chains[0]


class BlochFactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spring_system, cls.acoustic_sites = _spring_cell()

    def assert_cell_equations(self, system, result, mode):
        omega = math.tau * result.frequency_hz
        dynamic = system.K - omega**2 * system.M
        field = mode.displacement
        left, right = result.left_dofs, result.right_dofs
        interior = np.setdiff1d(np.arange(system.basis.N), np.r_[left, right])
        reaction = dynamic @ field
        scale = sparse_norm(dynamic) * np.linalg.norm(field)
        self.assertLess(np.linalg.norm(reaction[interior]) / scale, 1e-10)
        paired_reaction = reaction[left] + reaction[right] / mode.multiplier
        self.assertLess(np.linalg.norm(paired_reaction) / scale, 1e-10)
        np.testing.assert_allclose(field[right], mode.multiplier * field[left], rtol=0, atol=1e-10)
        # With exp(-i*omega*t), the right outward reaction gives positive axial flux.
        power = (omega / 2) * np.vdot(field[right], reaction[right]).imag
        np.testing.assert_allclose(mode.power_w, power, rtol=1e-10, atol=0)

    def test_chain_directions_follow_power_even_when_the_phase_is_folded(self):
        # q is the phase per spring. Each cell spans two springs, so q = 2*pi/3
        # has positive energy velocity but a negative principal phase per cell.
        for q in (math.pi / 3, 2 * math.pi / 3):
            with self.subTest(q=q):
                frequency = 2 * FREQUENCY_SCALE_HZ * math.sin(q / 2)
                result = solve_bloch_factors(self.spring_system, frequency)
                self.assertEqual(result.status, "resolved")
                self.assertEqual(len(result.modes), 2)
                self.assertAlmostEqual(result.period_m / PERIOD_M, 1.0)
                for sign, direction in ((1, "right"), (-1, "left")):
                    (mode,) = [mode for mode in result.modes if mode.direction == direction]
                    expected_phase = math.remainder(sign * 2 * q, math.tau)
                    self.assertAlmostEqual(mode.phase_rad, expected_phase, delta=1e-10)
                    self.assertAlmostEqual(mode.wave_number_rad_m * PERIOD_M, expected_phase, delta=1e-10)
                    acoustic_field = mode.displacement[self.acoustic_sites]
                    np.testing.assert_allclose(
                        acoustic_field / acoustic_field[0], np.exp(1j * sign * q * np.arange(3)),
                        rtol=0, atol=1e-10,
                    )
                    # Average power through a spring with the solved displacement amplitude.
                    expected_power = (
                        math.pi * frequency * SPRING_N_M * abs(acoustic_field[0])**2 * math.sin(sign * q)
                    )
                    np.testing.assert_allclose(mode.power_w, expected_power, rtol=1e-10, atol=0)
                    self.assert_cell_equations(self.spring_system, result, mode)

    def test_stopband_keeps_reciprocal_multipliers_without_propagating_modes(self):
        decay_per_spring = 0.4
        # The analytic continuation is q = pi + i*decay_per_spring.
        frequency = 2 * FREQUENCY_SCALE_HZ * math.cosh(decay_per_spring / 2)
        result = solve_bloch_factors(self.spring_system, frequency)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(len(result.modes), 0)
        alpha, beta = result.homogeneous_roots
        finite = beta != 0
        roots = alpha[finite] / beta[finite]
        for factor in (math.exp(-2 * decay_per_spring), math.exp(2 * decay_per_spring)):
            self.assertLess(np.min(abs(roots - factor)), 1e-10)

    def test_band_edge_does_not_label_zero_flux_as_propagating(self):
        result = solve_bloch_factors(
            self.spring_system, 2 * FREQUENCY_SCALE_HZ,
            unit_circle_tolerance=1e-6, relative_power_tolerance=1e-6,
        )
        self.assertEqual(result.status, "unresolved")
        self.assertTrue(result.modes)
        self.assertEqual({mode.direction for mode in result.modes}, {"unresolved"})

    def test_fixed_face_resonance_cannot_be_condensed(self):
        # At q = pi/2 the acoustic interior site has zero fixed-face dynamic stiffness.
        result = solve_bloch_factors(self.spring_system, math.sqrt(2) * FREQUENCY_SCALE_HZ)
        self.assertEqual(result.status, "unresolved")
        self.assertIsNone(result.boundary_dynamic_stiffness)
        self.assertEqual(result.homogeneous_roots.shape, (2, 0))
        self.assertEqual(result.modes, ())

    def test_three_dimensional_cell_matches_fixed_wavenumber_and_energy_flux(self):
        mesh = MeshTet.init_tensor(
            [0.0, PERIOD_M / 2, PERIOD_M], [-250e-9, 0.0, 250e-9], [-100e-9, 100e-9]
        )
        system = assemble_elasticity(mesh, CubicMaterial())
        wave_number = 0.57 / PERIOD_M
        reference = solve_bloch(system, wave_number, num_modes=4)
        branch = np.argmax(reference.polarization_fractions[:, 0])
        self.assertGreater(reference.polarization_fractions[branch, 0], 0.99)
        frequency = reference.frequencies_hz[branch]
        dk = 1e-5 / PERIOD_M
        f_lower = solve_bloch(system, wave_number - dk, num_modes=4).frequencies_hz[branch]
        f_upper = solve_bloch(system, wave_number + dk, num_modes=4).frequencies_hz[branch]
        group_velocity = math.tau * (f_upper - f_lower) / (2 * dk)
        self.assertGreater(group_velocity, 0)

        result = solve_bloch_factors(system, frequency)
        self.assertEqual(result.status, "resolved")
        for sign, direction in ((1, "right"), (-1, "left")):
            with self.subTest(direction=direction):
                phase = sign * 0.57
                mode = min(result.modes, key=lambda candidate: abs(candidate.phase_rad - phase))
                self.assertAlmostEqual(mode.phase_rad, phase, delta=1e-9)
                self.assertEqual(mode.direction, direction)
                target = reference.modes_full[:, branch]
                if sign < 0:
                    target = target.conj()
                mass_action = system.M @ mode.displacement
                modal_mass = np.vdot(mode.displacement, mass_action).real
                overlap = abs(np.vdot(target, mass_action)) / math.sqrt(modal_mass)
                self.assertAlmostEqual(overlap, 1.0, delta=1e-10)
                energy_per_length = (math.tau * frequency)**2 * modal_mass / (2 * PERIOD_M)
                np.testing.assert_allclose(
                    mode.power_w, sign * group_velocity * energy_per_length, rtol=2e-6, atol=0
                )
                self.assert_cell_equations(system, result, mode)


if __name__ == "__main__":
    unittest.main()
