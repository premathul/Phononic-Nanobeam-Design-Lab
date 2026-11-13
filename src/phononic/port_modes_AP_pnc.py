"""Lossless elastic waveguide ports. Author: Athul Prem | _AP_pnc.

The cross-section semi-analytical finite element (SAFE) method follows the
lossless formulation in Bartoli et al., doi:10.1016/j.jsv.2006.01.021.
"""

from dataclasses import dataclass
from math import isfinite, tau
from numbers import Integral, Real

import numpy as np
from scipy.linalg import cholesky, eig, eigh, solve_triangular
from skfem import Basis, BilinearForm, ElementTetP2, ElementTriP1, ElementTriP2, ElementVector, MeshTri
from skfem.helpers import ddot, dot

from .elasticity_AP_pnc import ElasticSystem
from .lead_modes_AP_pnc import _end_dof_pairs, _match_transverse_points
from .material_AP_pnc import CubicMaterial


@dataclass(frozen=True)
class PortSystem:
    """Shared prism lead model on the matching y,z triangular end faces.

    Each row matches one displacement DOF in both global face maps. ``K0``
    has units Pa, ``G`` N/m, ``Q`` N, and ``M`` kg/m. The inputs assume
    the same uniform material as the adjoining leads in the 3D body.
    """

    elastic_system: ElasticSystem
    basis: Basis
    left_dofs: np.ndarray
    right_dofs: np.ndarray
    K0: np.ndarray
    M: np.ndarray
    Q: np.ndarray
    G: np.ndarray
    reference_speed_m_s: float


@dataclass(frozen=True)
class PortModes:
    """Complete mode set of the discretized lossless lead at one frequency.

    Mode shapes are columns in the shared face basis. Forces use the +x
    traction on both ends. Positive/negative indices follow the sign of
    propagating power or direction of evanescent decay. Within either
    direction propagating modes come first, ordered by absolute wavenumber.
    Evanescent columns use a reference velocity to scale displacement.
    """

    port: PortSystem
    frequency_hz: float
    wavenumbers_rad_m: np.ndarray
    displacements: np.ndarray
    axial_forces: np.ndarray
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    propagating: np.ndarray
    powers_w: np.ndarray
    relative_residuals: np.ndarray

    def flux_gram(self, indices: np.ndarray) -> np.ndarray:
        """Return signed +x power as a Hermitian matrix in the chosen basis."""
        return _flux_gram(self.displacements[:, indices], self.axial_forces[:, indices], tau * self.frequency_hz)


def _transverse_gradient(u):
    # The two geometric coordinates of the face are lab y and z.
    gradient = np.zeros((3, 3, *u.grad.shape[2:]))
    gradient[:, 1:] = u.grad
    return gradient


def assemble_port(system: ElasticSystem, material: CubicMaterial, max_dofs: int = 200) -> PortSystem:
    """Assemble a homogeneous semi-infinite lead from congruent x end faces.

    A matched P1 or P2 triangular displacement trace retains three lab-frame
    components. The lead extends its section unchanged along x and has free
    lateral surfaces. Both body faces must have the same translated mesh.
    """
    if not isinstance(max_dofs, Integral) or isinstance(max_dofs, bool) or max_dofs < 1:
        raise ValueError("max_dofs must be a positive integer.")
    _, ordered_left, ordered_right = _end_dof_pairs(system)
    if len(ordered_left) > max_dofs:
        raise ValueError(f"Port has {len(ordered_left)} DOFs, exceeding dense solver cap {max_dofs}.")

    mesh = system.mesh
    boundary = mesh.boundary_facets()
    triangles = mesh.facets[:, boundary]
    x_min = mesh.p[0].min()
    atol = 64 * np.finfo(float).eps * max(abs(x_min), np.ptp(mesh.p[0]))
    triangles = triangles[:, np.all(abs(mesh.p[0, triangles] - x_min) <= atol, axis=0)]
    vertices = np.unique(triangles)
    section = MeshTri(mesh.p[1:, vertices], np.searchsorted(vertices, triangles))
    quadratic = isinstance(system.basis.elem.elem, ElementTetP2)
    element = ElementVector(ElementTriP2() if quadratic else ElementTriP1(), dim=3)
    basis = Basis(section, element, intorder=4 if quadratic else 2)

    left_dofs = np.empty(basis.N, dtype=int)
    right_dofs = np.empty_like(left_dofs)
    for transverse, volume in zip(basis.split_indices(), system.basis.split_indices()):
        selected = np.isin(ordered_left, volume)
        left = ordered_left[selected]
        mapped = _match_transverse_points(system.basis.doflocs[1:, left], basis.doflocs[:, transverse], np.ptp(section.p, axis=1))
        left_dofs[transverse] = left[mapped]
        right_dofs[transverse] = ordered_right[selected][mapped]

    stiffness = material.stiffness_tensor()

    # The minor symmetries of C permit gradient contractions in place of
    # symmetric strain contractions, including the x-derivative terms.
    @BilinearForm
    def stiffness_0(u, v, _):
        stress = np.einsum("ijkl,kl...->ij...", stiffness, _transverse_gradient(u))
        return ddot(_transverse_gradient(v), stress)

    @BilinearForm
    def axial_stiffness(u, v, _):
        return dot(v, np.einsum("ik,k...->i...", stiffness[:, 0, :, 0], u))

    @BilinearForm
    def coupling(u, v, _):
        stress_x = np.einsum("ikl,kl...->i...", stiffness[:, 0], _transverse_gradient(u))
        return dot(v, stress_x)

    @BilinearForm
    def mass(u, v, _):
        return material.rho * dot(u, v)

    K0 = stiffness_0.assemble(basis).toarray()
    Q = axial_stiffness.assemble(basis).toarray()
    return PortSystem(
        elastic_system=system,
        basis=basis,
        left_dofs=left_dofs,
        right_dofs=right_dofs,
        K0=(K0 + K0.T) / 2,
        M=mass.assemble(basis).toarray(),
        Q=(Q + Q.T) / 2,
        G=coupling.assemble(basis).toarray(),
        reference_speed_m_s=np.sqrt(material.C11 / material.rho),
    )


def _flux_gram(displacements: np.ndarray, forces: np.ndarray, omega: float) -> np.ndarray:
    work = displacements.conj().T @ forces
    return omega / (4j) * (work - work.conj().T)


def _scaled_eigenproblem(A: np.ndarray, H: np.ndarray, Q: np.ndarray, length: float):
    """Whiten Q/length² and solve for the eigenvalue i*k*length."""
    size = A.shape[0]
    lower = cholesky(Q / length**2, lower=True)

    def whiten(matrix):
        left_scaled = solve_triangular(lower, matrix, lower=True)
        return solve_triangular(lower, left_scaled.T, lower=True).T

    companion = np.block([[np.zeros_like(A), np.eye(size)], [whiten(A), whiten(H / length)]])
    values, left, right = eig(companion, left=True, right=True)
    # First-order eigenvalue sensitivity flags poorly resolved near-real and
    # near-zero wavenumbers. The tolerance is in dimensionless k*length.
    overlap = abs(np.sum(left.conj() * right, axis=0))
    matrix_norm = np.linalg.norm(companion, ord=np.inf)
    eps = np.finfo(float).eps
    uncertainty = 32 * eps * matrix_norm / np.maximum(overlap, eps)
    displacements = solve_triangular(lower.T, right[:size])
    displacements /= np.linalg.norm(displacements, axis=0)
    return -1j * values, displacements, uncertainty


def _orthogonalize_propagating(
    k_length, displacements, propagating, uncertainty, port, omega, length,
) -> None:
    """Diagonalize the signed power form within coincident real roots."""
    remaining = np.flatnonzero(propagating)
    while remaining.size:
        root = remaining[0]
        coincident = abs(k_length[remaining] - k_length[root]) <= np.maximum(uncertainty[remaining], uncertainty[root])
        indices = remaining[coincident]
        remaining = remaining[~coincident]
        if indices.size == 1:
            continue
        mean_root = np.mean(k_length[indices].real)
        rounding_tolerance = 1e-10 * max(1.0, abs(mean_root))
        if np.any(abs(k_length[indices] - mean_root) > rounding_tolerance):
            raise RuntimeError("Propagating wavenumbers are too close to distinguish numerically.")
        k_length[indices] = mean_root
        U = displacements[:, indices]
        F = (port.G + 1j * k_length[indices[0]] / length * port.Q) @ U
        # Mass whitening removes arbitrary conditioning of the dense
        # eigensolver's basis before diagonalizing signed axial power.
        mass_gram = U.conj().T @ (port.M @ U)
        _, rotation = eigh(_flux_gram(U, F, omega), mass_gram)
        U = U @ rotation
        displacements[:, indices] = U / np.linalg.norm(U, axis=0)


def _check_reciprocal_roots(k_length: np.ndarray, uncertainty: np.ndarray) -> None:
    # Reciprocity and real constitutive coefficients imply k -> -conj(k),
    # while the adjoint polynomial relation implies k -> conj(k).
    for partners in (-k_length.conj(), k_length.conj()):
        separation = abs(partners[:, None] - k_length[None, :])
        nearest = separation.argmin(axis=1)
        error = separation[np.arange(len(k_length)), nearest]
        if np.any(error > 8 * np.maximum(uncertainty, uncertainty[nearest])):
            raise RuntimeError("The SAFE roots failed the lossless reciprocal pairing check.")


def solve_port_modes(port: PortSystem, frequency_hz: float) -> PortModes:
    """Solve all 2N finite SAFE roots with exp(i*k*x - i*omega*t).

    Real roots are classified by signed energy flux and normalized to unit
    power. Nonreal roots are classified by decay direction. A numerically
    unresolved cutoff, singular displacement basis or failed flux check
    raises rather than returning an incomplete channel set.
    """
    if (not isinstance(frequency_hz, Real) or isinstance(frequency_hz, (bool, np.bool_))
            or not isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError("Port frequency must be positive and finite.")
    omega = tau * frequency_hz
    A = port.K0 - omega**2 * port.M
    H = port.G.T - port.G
    length = float(np.max(np.ptp(port.basis.mesh.p, axis=1)))
    k_length, U, uncertainty = _scaled_eigenproblem(A, H, port.Q, length)
    real_tolerance = 1e-10 * np.maximum(1.0, abs(k_length))
    tolerance = np.maximum(uncertainty, real_tolerance)
    attenuation = abs(k_length.imag)
    propagating = attenuation <= real_tolerance
    if np.any(abs(k_length) <= 8 * tolerance) or np.any(~propagating & (attenuation <= 8 * tolerance)):
        raise RuntimeError("Unresolved lead cutoff or weakly decaying root at this frequency.")
    _check_reciprocal_roots(k_length, tolerance)

    k_length[propagating] = k_length[propagating].real
    _orthogonalize_propagating(k_length, U, propagating, tolerance, port, omega, length)
    k = k_length / length
    forces = port.G @ U + (port.Q @ U) * (1j * k)
    imbalance = A @ U + (H @ U) * (1j * k) + (port.Q @ U) * k**2
    polynomial_norm = (
        np.linalg.norm(A, ord=np.inf) + abs(k) * np.linalg.norm(H, ord=np.inf)
        + abs(k)**2 * np.linalg.norm(port.Q, ord=np.inf)
    )
    residuals = np.linalg.norm(imbalance, axis=0) / (polynomial_norm * np.linalg.norm(U, axis=0))
    if not np.all(np.isfinite(residuals) & (residuals <= 1e-8)):
        raise RuntimeError(f"Port eigenpair backward residual exceeded 1e-8: {residuals.max():.3e}.")

    powers = omega / 2 * np.sum((U.conj() * forces).imag, axis=0)
    energy_per_length = omega**2 / 2 * np.sum((U.conj() * (port.M @ U)).real, axis=0)
    reference_power = energy_per_length * port.reference_speed_m_s
    relative_power = abs(powers) / reference_power
    if np.any(relative_power[propagating] <= 1e-8):
        raise RuntimeError("Unresolved lead cutoff or vanishing propagating-mode power.")

    direction = np.where(propagating, np.sign(powers), np.sign(k.imag))
    positive = np.flatnonzero(direction > 0)
    negative = np.flatnonzero(direction < 0)
    if positive.size != port.basis.N or negative.size != port.basis.N:
        raise RuntimeError("The two directional port bases are incomplete.")
    if np.count_nonzero(propagating[positive]) != np.count_nonzero(propagating[negative]):
        raise RuntimeError("The two directions have unequal propagating channel counts.")
    normalization = np.sqrt(np.where(propagating, abs(powers), reference_power))
    U /= normalization
    forces /= normalization

    def validate_and_order(indices, sign):
        order = np.lexsort((abs(k[indices]), ~propagating[indices]))
        indices = indices[order]
        directional = U[:, indices]
        condition = np.linalg.cond(directional / np.linalg.norm(directional, axis=0))
        if not np.isfinite(condition) or condition >= 1e10:
            raise RuntimeError(f"The directional displacement basis is ill-conditioned: {condition:.3e}.")
        flux = _flux_gram(directional, forces[:, indices], omega)
        flux_error = abs(flux - np.diag(sign * propagating[indices].astype(float))).max()
        if not np.isfinite(flux_error) or flux_error > 1e-7:
            raise RuntimeError(f"The directional axial power Gram error exceeded 1e-7 W: {flux_error:.3e} W.")
        return indices

    return PortModes(
        port=port,
        frequency_hz=float(frequency_hz),
        wavenumbers_rad_m=k,
        displacements=U,
        axial_forces=forces,
        positive_indices=validate_and_order(positive, 1),
        negative_indices=validate_and_order(negative, -1),
        propagating=propagating,
        powers_w=powers / normalization**2,
        relative_residuals=residuals,
    )
