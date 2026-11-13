"""Three-dimensional elastic finite elements. Author: Athul Prem | _AP_pnc."""

from dataclasses import dataclass
from math import tau

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from skfem import Basis, BilinearForm, Dofs, ElementTetP1, ElementTetP2, ElementVector, MeshTet
from skfem.helpers import ddot, dot, sym_grad

from .material_AP_pnc import CubicMaterial


@dataclass(frozen=True)
class ElasticSystem:
    """Unconstrained elastic system with coordinates in metres, K in N/m and M in kg."""

    mesh: MeshTet
    basis: Basis
    K: csr_matrix
    M: csr_matrix


@dataclass(frozen=True)
class ModeResult:
    """Finite-solid eigenmodes in ascending frequency order, including rigid modes.

    Columns of ``modes_full`` use the original DOF ordering, with unit modal
    mass and zeros at constrained DOFs. Polarization columns are the kinetic
    energy fractions of x, y, z motion. Rigid-mode fractions describe the
    numerical basis of their degenerate subspace.
    """

    frequencies_hz: np.ndarray
    modes_full: np.ndarray
    relative_residuals: np.ndarray
    free_dofs: np.ndarray
    polarization_fractions: np.ndarray
    rigid_mode_count: int


def assemble_elasticity(
    mesh: MeshTet, material: CubicMaterial, element_order: int = 2, max_dofs: int = 30_000
) -> ElasticSystem:
    """Assemble linear anisotropic stiffness and consistent mass on straight-sided tetrahedra.

    The mesh is one face-connected solid with coordinates in metres. Natural
    exterior boundaries are traction free. P1 and P2 refer to displacement
    interpolation. Degree-4 quadrature integrates both forms exactly for P2 on
    affine tetrahedra with uniform material.
    """
    if element_order not in (1, 2):
        raise ValueError("Tetrahedral displacement element order must be 1 or 2.")
    if not isinstance(mesh, MeshTet) or not mesh.affine or not mesh.nelements:
        raise ValueError("A nonempty, straight-sided tetrahedral mesh is required.")
    if not np.isfinite(mesh.p).all():
        raise ValueError("Mesh coordinates must be finite.")

    element = ElementVector(ElementTetP1() if element_order == 1 else ElementTetP2())
    dofs = Dofs(mesh, element)
    if dofs.N > max_dofs:
        raise ValueError(f"Mesh has {dofs.N} displacement DOFs, exceeding cap {max_dofs}.")

    neighbors = mesh.f2t[:, mesh.f2t[1] >= 0]
    adjacency = csr_matrix(
        (np.ones(neighbors.shape[1]), (neighbors[0], neighbors[1])),
        shape=(mesh.nelements, mesh.nelements),
    )
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError("Mesh tetrahedra must form one face-connected solid.")
    determinants = mesh.mapping().detA
    if np.any(determinants == 0) or not np.isfinite(determinants).all():
        raise ValueError("Mesh contains a degenerate tetrahedron.")

    basis = Basis(mesh, element, intorder=4 if element_order == 2 else 2, dofs=dofs)
    stiffness = material.stiffness_tensor()

    @BilinearForm
    def elastic_form(u, v, _):
        stress = np.einsum("ijkl,kl...->ij...", stiffness, sym_grad(u))
        return ddot(stress, sym_grad(v))

    @BilinearForm
    def mass_form(u, v, _):
        return material.rho * dot(u, v)

    K = elastic_form.assemble(basis)
    M = mass_form.assemble(basis)
    return ElasticSystem(mesh=mesh, basis=basis, K=(K + K.T) * 0.5, M=M)


def solve_generalized(
    K: csr_matrix, M: csr_matrix, num_modes: int, rigid_mode_count: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the lowest eigenpairs of the real or complex Hermitian pencil K−omega² M.

    The inputs must be a compatible positive-semidefinite stiffness and
    positive-definite mass, with a known number of rigid modes. Output is
    omega² in s⁻², vectors as columns with unit modal mass, and normwise
    backward residuals. The requested count includes any rigid modes.
    """
    size = K.shape[0]
    if not 0 <= rigid_mode_count <= num_modes <= size or num_modes < 1:
        raise ValueError("Require 0 <= rigid_mode_count <= num_modes <= matrix size.")

    k_scale = float(np.max(K.diagonal().real))
    m_scale = float(np.max(M.diagonal().real))
    if not k_scale > 0 or not m_scale > 0:
        raise ValueError("Stiffness and mass must have positive characteristic diagonals.")
    scaled_K = K / k_scale
    scaled_M = M / m_scale

    # Inspect at least one elastic eigenvalue when returning only rigid modes.
    count = max(num_modes, min(size, rigid_mode_count + 1))
    if size <= 256 or count >= size - 1:
        values, vectors = eigh(scaled_K.toarray(), scaled_M.toarray(), subset_by_index=(0, count - 1))
    else:
        # A negative shift keeps K - sigma M invertible even for a free solid.
        # It sets the eigensolver's search point only and leaves the pencil intact.
        _, candidates = eigsh(
            scaled_K, k=count, M=scaled_M, sigma=-1e-4, which="LM", tol=1e-11,
            ncv=min(size, max(32, 3 * count + 1)), rng=np.random.default_rng(12345),
        )
        # Complex ARPACK eigenvectors can be nonorthogonal inside a degenerate
        # subspace. Re-diagonalize both forms in the computed subspace.
        projected_K = candidates.conj().T @ (scaled_K @ candidates)
        projected_M = candidates.conj().T @ (scaled_M @ candidates)
        values, rotation = eigh(
            (projected_K + projected_K.conj().T) * 0.5,
            (projected_M + projected_M.conj().T) * 0.5,
        )
        vectors = candidates @ rotation

    k_norm = float(np.asarray(abs(scaled_K).sum(axis=1)).max())
    m_norm = float(np.asarray(abs(scaled_M).sum(axis=1)).max())
    mass_action = scaled_M @ vectors
    vector_norm = np.linalg.norm(vectors, axis=0)
    # Estimate how a zero frequency is resolved relative to matrix roundoff.
    zero_residual = abs(values) * np.linalg.norm(mass_action, axis=0) / (k_norm * vector_norm)
    zero_tolerance = 256 * np.finfo(float).eps
    if np.any(zero_residual[:rigid_mode_count] > zero_tolerance):
        raise RuntimeError("The requested rigid modes did not have numerically zero frequencies.")
    if np.any(values[rigid_mode_count:] <= 0) or (
        rigid_mode_count and np.any(zero_residual[rigid_mode_count:] <= zero_tolerance)
    ):
        raise RuntimeError("The computed elastic frequencies are nonpositive or unresolved from rigid modes.")

    values[:rigid_mode_count] = 0.0
    imbalance = scaled_K @ vectors - mass_action * values
    residuals = np.linalg.norm(imbalance, axis=0) / ((k_norm + abs(values) * m_norm) * vector_norm)
    if not np.all(np.isfinite(residuals) & (residuals <= 1e-8)):
        raise RuntimeError(f"Eigenpair backward residual exceeded 1e-8: {residuals.max():.3e}.")
    return (
        values[:num_modes] * (k_scale / m_scale),
        vectors[:, :num_modes] / np.sqrt(m_scale),
        residuals[:num_modes],
    )


def polarization_fractions(system: ElasticSystem, modes_full: np.ndarray) -> np.ndarray:
    """Return each component's share of the consistent-mass quadratic form."""
    mass_action = system.M @ modes_full
    fractions = np.column_stack([
        np.sum((modes_full[component].conj() * mass_action[component]).real, axis=0)
        for component in system.basis.split_indices()
    ])
    return fractions / fractions.sum(axis=1, keepdims=True)


def _clamp_dofs(system: ElasticSystem, label: str, end_x: float) -> np.ndarray:
    mesh = system.mesh
    facets = (mesh.boundaries or {}).get(label)
    if facets is None:
        boundary_facets = mesh.boundary_facets()
        vertices_x = mesh.p[0, mesh.facets[:, boundary_facets]]
        atol = 64 * np.finfo(float).eps * max(abs(end_x), np.ptp(mesh.p[0]))
        facets = boundary_facets[np.all(abs(vertices_x - end_x) <= atol, axis=0)]
    if len(facets) == 0 or not np.isin(facets, mesh.boundary_facets()).all():
        raise ValueError(f"Clamp {label} requires a nonempty exterior face set.")
    return system.basis.get_dofs(facets=facets).all()


def solve_modes(system: ElasticSystem, boundary: str = "clamped_free", num_modes: int = 8) -> ModeResult:
    """Solve free vibration with free, left-clamped, or both-ends-clamped boundaries.

    Clamps use exterior face tags ``xleft`` and ``xright`` when present, or
    planar x-extremal faces otherwise. A connected free solid has six rigid
    modes, included in the requested mode count when using ``boundary='free'``.
    """
    dofs = np.arange(system.basis.N)
    rigid_mode_count = 0
    if boundary == "free":
        free_dofs = dofs
        rigid_mode_count = 6
    elif boundary in ("clamped_free", "clamped_clamped"):
        x = system.mesh.p[0]
        clamps = _clamp_dofs(system, "xleft", x.min())
        if boundary == "clamped_clamped":
            clamps = np.union1d(clamps, _clamp_dofs(system, "xright", x.max()))
        free_dofs = np.setdiff1d(dofs, clamps, assume_unique=True)
    else:
        raise ValueError(f"Unsupported boundary condition {boundary!r}.")

    if num_modes < 1:
        raise ValueError("At least one mode must be requested.")
    K = system.K[free_dofs][:, free_dofs]
    M = system.M[free_dofs][:, free_dofs]
    values, modes, residuals = solve_generalized(K, M, max(num_modes, rigid_mode_count), rigid_mode_count)
    modes_full = np.zeros((system.basis.N, num_modes), dtype=modes.dtype)
    modes_full[free_dofs, :] = modes[:, :num_modes]
    return ModeResult(
        frequencies_hz=np.sqrt(values[:num_modes]) / tau,
        modes_full=modes_full,
        relative_residuals=residuals[:num_modes],
        free_dofs=free_dofs,
        polarization_fractions=polarization_fractions(system, modes_full),
        rigid_mode_count=min(rigid_mode_count, num_modes),
    )
