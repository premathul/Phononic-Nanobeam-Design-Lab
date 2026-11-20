"""Finite joined-cell response checks. Author: Athul Prem | _AP_pnc."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from skfem import MeshTet

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import solve_scattering
from tools.bloch_factors_AP_pnc import solve_bloch_factors
from tools.finite_mirror_response_AP_pnc import compare_native_response, solve_finite_mirror_response


class FiniteMirrorResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.period = 600e-9
        cls.frequency = 300e6
        cls.material = CubicMaterial()

    def uniform_cell_chain(self, count):
        mesh = MeshTet.init_tensor(
            np.linspace(0, count * self.period, 2 * count + 1),
            [-250e-9, 0.0, 250e-9], [-100e-9, 100e-9],
        )
        return assemble_elasticity(mesh, self.material)

    def test_join_reproduces_native_original_plane_response_for_one_and_two_cells(self):
        system = self.uniform_cell_chain(1)
        condensation = solve_bloch_factors(system, self.frequency)
        native_port = assemble_port(system, self.material)
        size = native_port.basis.N

        # Persist every block and mass in a different order than the native SAFE basis.
        order = np.arange(size)[::-1]
        dofs = condensation.left_dofs[order]
        native_positions = {dof: index for index, dof in enumerate(native_port.left_dofs)}
        native_order = [native_positions[dof] for dof in dofs]
        components = np.empty(size, dtype=int)
        for component, indices in enumerate(native_port.basis.split_indices()):
            components[indices] = component
        blocks = condensation.boundary_dynamic_stiffness[np.ix_(np.r_[order, order + size], np.r_[order, order + size])]

        with TemporaryDirectory() as temporary:
            archive = Path(temporary) / "traces_AP_pnc.npz"
            np.savez(
                archive, frequency_hz=self.frequency, period_m=self.period,
                trace_dof_yz_m=system.basis.doflocs[1:, dofs].T, trace_dof_components=components[native_order],
                trace_mass=native_port.M[np.ix_(native_order, native_order)],
                LL=blocks[:size, :size], LR=blocks[:size, size:], RL=blocks[size:, :size], RR=blocks[size:, size:],
            )
            for count in (1, 2):
                with self.subTest(count=count):
                    reference = system if count == 1 else self.uniform_cell_chain(count)
                    modes = solve_port_modes(assemble_port(reference, self.material), self.frequency)
                    native = solve_scattering(reference, modes, self.frequency)
                    result = solve_finite_mirror_response(archive, modes, count)
                    json.dumps(result, allow_nan=False)
                    comparison = compare_native_response(result, native)
                    self.assertEqual(comparison["status"], "accepted")
                    self.assertLess(max(comparison["maximum_absolute_changes"].values()), 1e-10)
                    self.assertEqual(result["status"], "accepted")
                    self.assertEqual(result["reference_planes_m"], [0, count * self.period])
                    self.assertGreater(result["native_port"]["evanescent_columns_per_end"], 0)
                    self.assertLess(max(result["relative_residuals"].values()), 1e-10)
                    self.assertLess(max(result["power_errors"].values()), 1e-10)


if __name__ == "__main__":
    unittest.main()
