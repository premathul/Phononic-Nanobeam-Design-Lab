"""Element-batched elastic assembly. Author: Athul Prem | _AP_pnc."""

from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skfem import Basis, BilinearForm, Dofs, ElementTetP1, ElementTetP2, ElementVector, MeshTet
from skfem.assembly.basis import AbstractBasis
from skfem.helpers import ddot, dot, sym_grad

from .elasticity_AP_pnc import ElasticSystem
from .material_AP_pnc import CubicMaterial


class _DisplacementDofLayout(AbstractBasis):
    """Global DOF numbering and locations without evaluated cell fields."""

    def __init__(self, mesh: MeshTet, element: ElementVector, dofs: Dofs, intorder: int):
        super().__init__(mesh, element, intorder=intorder, refdom=mesh.refdom, dofs=dofs)

    def interpolate(self, *_args, **_kwargs):
        raise NotImplementedError("The displacement DOF layout has no evaluated cell fields.")


def assemble_elasticity_batched(
    mesh: MeshTet,
    material: CubicMaterial,
    element_order: int = 2,
    max_dofs: int = 30_000,
    batch_elements: int = 2048,
) -> ElasticSystem:
    """Assemble the SI stiffness and consistent mass using bounded cell batches.

    P1 and P2 on affine tetrahedra use the same degree-2 and degree-4 rules as
    the full-basis assembler. All batches use the original global DOF layout.
    The returned system retains the layout for solves and export, without a
    volume quadrature cache. Sparse matrices still grow with the global mesh.
    """
    if (not isinstance(batch_elements, Integral) or isinstance(batch_elements, bool)
            or batch_elements < 1):
        raise ValueError("batch_elements must be a positive integer.")
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

    stiffness = material.stiffness_tensor()

    @BilinearForm
    def elastic_form(u, v, _):
        stress = np.einsum("ijkl,kl...->ij...", stiffness, sym_grad(u))
        return ddot(stress, sym_grad(v))

    @BilinearForm
    def mass_form(u, v, _):
        return material.rho * dot(u, v)

    intorder = 4 if element_order == 2 else 2
    K = csr_matrix((dofs.N, dofs.N))
    M = csr_matrix(K.shape)
    batch_elements = int(batch_elements)
    for start in range(0, mesh.nelements, batch_elements):
        cells = np.arange(start, min(start + batch_elements, mesh.nelements))
        batch = Basis(mesh, element, intorder=intorder, elements=cells, dofs=dofs, disable_doflocs=True)
        K = K + elastic_form.assemble(batch)
        M = M + mass_form.assemble(batch)
        del batch

    layout = _DisplacementDofLayout(mesh, element, dofs, intorder)
    return ElasticSystem(mesh=mesh, basis=layout, K=(K + K.T) * 0.5, M=M)
