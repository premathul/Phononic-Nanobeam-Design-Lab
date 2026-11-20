"""Regional work on the native affine tetrahedral P2 basis. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

from dataclasses import dataclass
from math import tau
from numbers import Integral, Real

import numpy as np
from skfem import Dofs, ElementTetP2, ElementVector, MeshTet
from skfem.quadrature import get_quadrature


@dataclass(frozen=True)
class RegionalBilinearWork:
    """Group labels and contractions ordered as (group, dual column, primal column)."""

    groups: np.ndarray
    stiffness: np.ndarray
    inertia: np.ndarray

    @property
    def dynamic(self) -> np.ndarray:
        """Return the stiffness minus the omega-squared mass contraction."""
        return self.stiffness - self.inertia


def _evaluate_fields(
    field: np.ndarray,
    cell_dofs: np.ndarray,
    shapes: np.ndarray,
    gradients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate displacement and symmetric strain for every column at quadrature."""
    n_cells, n_quadrature, n_columns = len(cell_dofs), len(shapes), field.shape[1]
    coefficients = field[cell_dofs].reshape(n_cells, shapes.shape[1], 3 * n_columns)
    values = (shapes @ coefficients).reshape(n_cells, n_quadrature, 3, n_columns)
    gradient = (gradients @ coefficients).reshape(n_cells, n_quadrature, 3, 3, n_columns)
    strain = 0.5 * (gradient + gradient.swapaxes(2, 3))
    return values, strain.reshape(n_cells, n_quadrature, 9, n_columns)


def regional_bilinear_work(
    mesh: MeshTet,
    material,
    u: np.ndarray,
    z: np.ndarray,
    element_groups: np.ndarray,
    frequency_hz: float,
    *,
    batch_elements: int = 1024,
) -> RegionalBilinearWork:
    """Compute z.T K_group u and z.T (omega² M_group) u over whole tetrahedra.

    Columns in u and z follow the native scikit-fem vector P2 DOFs. The
    bilinear product uses no complex conjugation. Mesh coordinates and
    material properties are SI, and frequency is in Hz. Group labels are
    sorted and must give one integer or text label per tetrahedron.
    The integrations cover volume terms only, including cells adjacent to
    the ends. Port and excitation terms are outside these contractions.
    """
    if not isinstance(mesh, MeshTet) or not mesh.affine or not mesh.nelements:
        raise ValueError("A nonempty, straight-sided tetrahedral mesh is required.")
    if not np.isfinite(mesh.p).all():
        raise ValueError("Mesh coordinates must be finite.")
    if not isinstance(batch_elements, Integral) or isinstance(batch_elements, bool) or batch_elements < 1:
        raise ValueError("batch_elements must be a positive integer.")
    if not isinstance(frequency_hz, Real) or isinstance(frequency_hz, bool) or frequency_hz < 0:
        raise ValueError("frequency_hz must be a finite, nonnegative real number.")
    omega = tau * float(frequency_hz)
    omega_squared = omega * omega
    if not np.isfinite(omega_squared):
        raise ValueError("Squared angular frequency must be finite.")

    groups = np.asarray(element_groups)
    if groups.shape != (mesh.nelements,) or groups.dtype.kind not in "iuUS":
        raise ValueError("Assign one integer or text group label to each tetrahedron.")
    labels, group_index = np.unique(groups, return_inverse=True)

    scalar_element = ElementTetP2()
    dofs = Dofs(mesh, ElementVector(scalar_element))
    u, z = np.asarray(u), np.asarray(z)
    for field in (u, z):
        if (field.ndim != 2 or field.shape[0] != dofs.N or field.shape[1] == 0
                or field.dtype.kind not in "iufc" or not np.isfinite(field).all()):
            raise ValueError("u and z must contain finite columns in the native vector P2 DOF order.")
    output_shape = (len(labels), z.shape[1], u.shape[1])
    stiffness = np.zeros(output_shape, dtype=np.result_type(u, z, np.float64))
    inertia = np.zeros_like(stiffness)

    # ElementVector interleaves x, y, z components for each local scalar shape.
    # Use the same H1 mapping and degree-4 rule as the native vector P2 assembly.
    points, weights = get_quadrature(scalar_element, 4)
    n_shapes = scalar_element.doflocs.shape[0]
    n_quadrature = len(weights)
    reference_shapes = [scalar_element.lbasis(points, i) for i in range(n_shapes)]
    shapes = np.stack([value for value, _ in reference_shapes], axis=1)
    reference_gradients = np.stack([gradient for _, gradient in reference_shapes])
    mapping = mesh.mapping()
    stiffness_tensor = material.stiffness_tensor().reshape(9, 9)

    # Budget for geometry, field and pair arrays with room for temporary
    # copies. Global DOFs and the mesh mapping are shared between batches.
    n_columns = u.shape[1] + z.shape[1]
    n_pairs = u.shape[1] * z.shape[1]
    geometry_values = n_quadrature * (9 + 3 * n_shapes + 2)
    field_values = n_columns * (3 * n_shapes + n_quadrature * (3 + 9))
    bytes_per_cell = 4 * stiffness.dtype.itemsize * (geometry_values + field_values + n_pairs)
    max_batch = (120 * 1024**2) // bytes_per_cell
    batch_elements = min(batch_elements, max(1, max_batch))
    for start in range(0, mesh.nelements, batch_elements):
        stop = min(start + batch_elements, mesh.nelements)
        cells = np.arange(start, stop)
        determinants = mapping.detDF(points, cells)
        if np.any(determinants == 0) or not np.isfinite(determinants).all():
            raise ValueError("Mesh contains a degenerate tetrahedron.")
        integration_weights = abs(determinants) * weights
        inverse_jacobians = mapping.invDF(points, cells)
        gradients = np.einsum("jkeq,ajq->eqka", inverse_jacobians, reference_gradients, order="C")
        gradients = gradients.reshape(len(cells), n_quadrature * 3, n_shapes)
        cell_dofs = dofs.element_dofs[:, start:stop].T
        u_values, u_strain = _evaluate_fields(u, cell_dofs, shapes, gradients)
        z_values, z_strain = _evaluate_fields(z, cell_dofs, shapes, gradients)

        stress = stiffness_tensor @ u_strain
        stress *= integration_weights[..., None, None]
        cell_stiffness = np.einsum("eqin,eqim->enm", z_strain, stress)
        u_values *= (integration_weights * (omega_squared * material.rho))[..., None, None]
        cell_inertia = np.einsum("eqin,eqim->enm", z_values, u_values)
        batch_groups = group_index[start:stop]
        for group in np.unique(batch_groups):
            selected = batch_groups == group
            stiffness[group] += cell_stiffness[selected].sum(axis=0)
            inertia[group] += cell_inertia[selected].sum(axis=0)

    return RegionalBilinearWork(labels, stiffness, inertia)
