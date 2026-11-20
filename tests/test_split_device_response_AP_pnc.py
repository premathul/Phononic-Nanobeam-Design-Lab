"""Exact split of a small nonuniform perforated device. Athul Prem | _AP_pnc."""

from importlib.util import find_spec
from math import tau
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import get_blas_funcs

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, Hole
from phononic.layered_AP_pnc import make_layered_mesh
from phononic.linear_lgmres_AP_pnc import _original_residual
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import solve_scattering
from tools.split_device_response_AP_pnc import build_axial_partition, solve_partitioned_operator


@unittest.skipUnless(find_spec("gmsh"), "requires gmsh")
class SplitDeviceResponseTests(unittest.TestCase):
    def test_nonplanar_split_preserves_original_field_traction_and_response(self):
        frequency = 0.8875e9
        geometry = BeamGeometry(1600e-9, 420e-9, 200e-9)
        holes = (Hole(560e-9, 160e-9), Hole(1020e-9, 180e-9))
        material = CubicMaterial()
        mesh = make_layered_mesh(geometry, holes, 180e-9, port_mesh_size_m=240e-9,
                                 thickness_layers=1, circle_points=12)
        system = assemble_elasticity(mesh, material, max_dofs=5000)
        port = assemble_port(system, material)
        modes = solve_port_modes(port, frequency)

        cuts = (310e-9, 790e-9, 1270e-9)
        partition = build_axial_partition(system, cuts, protected_interval_m=(cuts[0], cuts[-1]))
        self.assertTrue(all(dofs.size > 0 for dofs in partition.interior_dofs))
        facet_x = mesh.p[0, mesh.facets[:, partition.frontier_facets]]
        self.assertGreater(np.max(np.ptp(facet_x, axis=0)), 1e-9)
        element_x = mesh.p[0, mesh.t]
        intersects = (element_x.max(axis=0) >= cuts[0]) & (element_x.min(axis=0) <= cuts[-1])
        self.assertTrue(np.isin(partition.element_regions[intersects], (1, 2)).all())
        shared_pairs = mesh.f2t[:, partition.frontier_facets]
        self.assertTrue(np.all(partition.element_regions[shared_pairs[0]] != partition.element_regions[shared_pairs[1]]))

        dynamic = system.K - (tau * frequency)**2 * system.M
        end_dofs = np.r_[port.left_dofs, port.right_dofs]
        unit_power_fields = []
        for side in ("left", "right"):
            with self.subTest(incident_side=side):
                report = {}

                def linear_solver(operator, rhs):
                    solution = solve_partitioned_operator(operator, rhs, partition, rhs_block_size=3, report=report)
                    nrm2 = get_blas_funcs("nrm2", (rhs,))
                    norm = float(np.asarray(abs(operator).sum(axis=1)).max())
                    for x, b in zip(solution.T, rhs.T, strict=True):
                        relative, backward, ok = _original_residual(
                            operator, x, b, nrm2, nrm2(b), float(np.max(abs(b))), norm, 1e-8, {},
                        )
                        self.assertTrue(ok, (relative, backward))
                    return solution

                direct = solve_scattering(system, modes, frequency, side)
                split = solve_scattering(system, modes, frequency, side, linear_solver=linear_solver)
                self.assertEqual(report["separator_dofs"], partition.separator_dofs.size)
                self.assertLess(report["separator_dofs"], system.basis.N)
                self.assertTrue(all(row["maximum_interface_rhs_columns"] == 3 for row in report["regions"]))
                self.assertEqual([row["interior_dtype"] for row in report["regions"]],
                                 ["complex128", "float64", "float64", "complex128"])
                for name in ("incident_mode_indices", "reflection_all_mode_indices", "transmission_all_mode_indices"):
                    np.testing.assert_array_equal(getattr(split, name), getattr(direct, name))
                for name in ("reflection_amplitudes_full", "transmission_amplitudes_full", "reflection", "transmission",
                             "reflection_flux", "transmission_flux", "reflection_gram", "transmission_gram"):
                    np.testing.assert_allclose(getattr(split, name), getattr(direct, name), atol=1e-8, rtol=0)
                np.testing.assert_allclose(split.power_balance, 0, atol=1e-8, rtol=0)
                self.assertLess(np.max(split.relative_residuals), 1e-8)

                scale = np.max(abs(direct.displacements), axis=0)
                self.assertLess(np.max(abs(split.displacements - direct.displacements) / scale), 1e-8)
                direct_force = (dynamic @ direct.displacements)[end_dofs]
                split_force = (dynamic @ split.displacements)[end_dofs]
                self.assertLess(np.max(abs(split_force - direct_force) / np.max(abs(direct_force), axis=0)), 1e-8)
                unit_power_fields.append(split)

        def native_scattering_columns(result, outgoing_amps, outgoing_ids):
            return np.sqrt(abs(modes.powers_w[outgoing_ids, None])) * outgoing_amps / np.sqrt(result.incident_powers_w)

        left, right = unit_power_fields
        scattering = np.block([
            [native_scattering_columns(left, left.reflection_amplitudes, left.reflection_mode_indices),
             native_scattering_columns(right, right.transmission_amplitudes, right.transmission_mode_indices)],
            [native_scattering_columns(left, left.transmission_amplitudes, left.transmission_mode_indices),
             native_scattering_columns(right, right.reflection_amplitudes, right.reflection_mode_indices)],
        ])
        np.testing.assert_allclose(scattering.conj().T @ scattering, np.eye(scattering.shape[0]), atol=1e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()
