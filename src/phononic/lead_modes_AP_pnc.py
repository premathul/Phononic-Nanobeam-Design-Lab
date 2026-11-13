"""Three-dimensional Bloch dispersion of an elastic guide. Author: Athul Prem | _AP_pnc."""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .elasticity_AP_pnc import ElasticSystem, polarization_fractions, solve_generalized


@dataclass(frozen=True)
class BlochResult:
    """Sampled eigenpairs ordered by frequency, with polarization columns x, y, z.

    The full-cell vectors are columns in original basis order and are mass normalized.
    Residuals are evaluated in the constrained space, where the boundary work cancels.
    The polarization fractions measure component shares of the kinetic energy.
    """

    k_rad_m: float
    period_m: float
    frequencies_hz: np.ndarray
    modes_full: np.ndarray
    relative_residuals: np.ndarray
    polarization_fractions: np.ndarray
    rigid_mode_count: int
    reduced_dof_count: int


def _match_transverse_points(
    left: np.ndarray, right: np.ndarray, transverse_span: np.ndarray
) -> np.ndarray:
    if left.shape != right.shape or left.shape[1] == 0:
        raise ValueError("Bloch end faces must have matching nonempty point sets.")
    # Match in normalized transverse coordinates for the nanometer-scale cell.
    distance, indices = cKDTree((left / transverse_span[:, None]).T).query(
        (right / transverse_span[:, None]).T
    )
    if np.any(distance > 1e-10) or np.unique(indices).size != indices.size:
        raise ValueError("Bloch end-face points must match one-to-one under x translation.")
    return indices


def _end_dof_pairs(system: ElasticSystem) -> tuple[float, np.ndarray, np.ndarray]:
    mesh, basis = system.mesh, system.basis
    span = np.ptp(mesh.p, axis=1)
    period = float(span[0])
    x_left, x_right = mesh.p[0].min(), mesh.p[0].max()
    x_atol = 64 * np.finfo(float).eps * max(abs(x_left), abs(x_right), period)
    if period <= 2 * x_atol or np.any(span[1:] <= 0):
        raise ValueError("Bloch dispersion requires a three-dimensional cell of positive length.")

    boundary = mesh.boundary_facets()
    boundary_x = mesh.p[0, mesh.facets[:, boundary]]
    left_facets = boundary[np.all(np.abs(boundary_x - x_left) <= x_atol, axis=0)]
    right_facets = boundary[np.all(np.abs(boundary_x - x_right) <= x_atol, axis=0)]
    left_triangles, right_triangles = mesh.facets[:, left_facets], mesh.facets[:, right_facets]
    left_vertices, right_vertices = np.unique(left_triangles), np.unique(right_triangles)
    vertex_pairs = _match_transverse_points(
        mesh.p[1:, left_vertices], mesh.p[1:, right_vertices], span[1:]
    )
    mapped_right = left_vertices[vertex_pairs[np.searchsorted(right_vertices, right_triangles)]]
    left_faces = {tuple(sorted(triangle)) for triangle in left_triangles.T}
    right_faces = {tuple(sorted(triangle)) for triangle in mapped_right.T}
    if left_faces != right_faces:
        raise ValueError("Bloch end faces must have matching triangulations.")

    left_face_dofs = basis.get_dofs(facets=left_facets).all()
    right_face_dofs = basis.get_dofs(facets=right_facets).all()
    paired_left, paired_right = [], []
    for component in basis.split_indices():
        left = np.intersect1d(left_face_dofs, component, assume_unique=True)
        right = np.intersect1d(right_face_dofs, component, assume_unique=True)
        dof_pairs = _match_transverse_points(
            basis.doflocs[1:, left], basis.doflocs[1:, right], span[1:]
        )
        paired_left.append(left[dof_pairs])
        paired_right.append(right)
    return period, np.concatenate(paired_left), np.concatenate(paired_right)


def solve_bloch(system: ElasticSystem, k_rad_m: float, num_modes: int = 8) -> BlochResult:
    """Solve real-wavenumber dispersion for a connected x-periodic solid with free sides.

    The two planar x end faces must share a translated mesh. For the exp(-i*omega*t)
    convention, right-face displacement equals exp(i*k*period) times left-face
    displacement. At phase 1 the connected cell admits three rigid translations
    and a rigid rotation about x. Returned branches are a finite requested sample,
    which may include fewer than all four rigid modes.
    """
    if not isinstance(num_modes, Integral) or isinstance(num_modes, bool) or num_modes < 1:
        raise ValueError("num_modes must be a positive integer.")
    period, left, right = _end_dof_pairs(system)
    phase_angle = k_rad_m * period
    if not math.isfinite(phase_angle):
        raise ValueError("Bloch wavenumber and the resulting phase must be finite.")
    phase_angle = math.remainder(phase_angle, math.tau)
    if phase_angle == 0:
        phase = 1.0
    elif abs(phase_angle) == math.pi:
        phase = -1.0
    else:
        phase = np.exp(1j * phase_angle)

    full_dofs = np.arange(system.basis.N)
    independent = np.ones(system.basis.N, dtype=bool)
    independent[right] = False
    reduced_dof_count = int(independent.sum())
    columns = np.empty(system.basis.N, dtype=int)
    columns[independent] = np.arange(reduced_dof_count)
    columns[right] = columns[left]
    weights = np.ones(system.basis.N, dtype=type(phase))
    weights[right] = phase
    projection = csr_matrix(
        (weights, (full_dofs, columns)), shape=(system.basis.N, reduced_dof_count)
    )
    test_projection = projection.getH()
    K = (test_projection @ system.K @ projection).tocsr()
    M = (test_projection @ system.M @ projection).tocsr()
    rigid_mode_count = 4 if phase_angle == 0 else 0
    eigenvalues, reduced_modes, residuals = solve_generalized(
        K, M, max(num_modes, rigid_mode_count), rigid_mode_count=rigid_mode_count
    )
    modes_full = projection @ reduced_modes[:, :num_modes]
    return BlochResult(
        k_rad_m=float(k_rad_m),
        period_m=period,
        frequencies_hz=np.sqrt(eigenvalues[:num_modes]) / math.tau,
        modes_full=modes_full,
        relative_residuals=residuals[:num_modes],
        polarization_fractions=polarization_fractions(system, modes_full),
        rigid_mode_count=rigid_mode_count,
        reduced_dof_count=reduced_dof_count,
    )
