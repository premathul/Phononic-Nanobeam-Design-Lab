"""Nominal cubic crystal elasticity in SI units. Author: Athul Prem | _AP_pnc."""

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

import numpy as np


def _is_real_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True, eq=False)
class CubicMaterial:
    """Cubic stiffnesses in Pa, density in kg/m³, and crystal-to-lab rotation.

    Defaults are the project's historical fixed-property Ge inputs, with the
    cubic axes aligned to lab x, y, z. Their temperature validity is unestablished.
    Columns of ``rotation`` are the crystal axes expressed in lab coordinates.
    """

    C11: float = 128.5e9
    C12: float = 48.3e9
    C44: float = 66.8e9
    rho: float = 5323.0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        for name in ("C11", "C12", "C44", "rho"):
            value = getattr(self, name)
            if not _is_real_number(value):
                raise ValueError(f"{name} must be a finite real number.")
            try:
                value = float(value)
            except OverflowError as error:
                raise ValueError(f"{name} must be a finite real number.") from error
            if not isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
            object.__setattr__(self, name, value)
        if self.rho <= 0:
            raise ValueError("Density must be positive.")
        if min(self.C44, self.C11 - self.C12, self.C11 + 2 * self.C12) <= 0:
            raise ValueError("Cubic stiffnesses must give positive elastic energy.")

        # Preserve the element types of mixed JSON lists until they are checked.
        rotation = np.asarray(self.rotation, dtype=object)
        if rotation.shape != (3, 3) or not all(_is_real_number(value) for value in rotation.flat):
            raise ValueError("Rotation must be a real 3 × 3 matrix.")
        try:
            rotation = np.array(rotation, dtype=float)
        except OverflowError as error:
            raise ValueError("Rotation entries must be finite.") from error
        if not np.isfinite(rotation).all() or not (
            np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10)
            and np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-10)
        ):
            raise ValueError("Rotation must be orthogonal with determinant +1.")
        rotation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)

    def stiffness_tensor(self) -> np.ndarray:
        """Return lab-frame C[i,j,k,l] for sigma[i,j] = C[i,j,k,l] epsilon[k,l].

        ``epsilon`` is the symmetric strain tensor. Summation over both xy and
        yx gives sigma_xy = 2 C44 epsilon_xy in the crystal frame.
        """
        identity = np.eye(3)
        axes = self.rotation
        cubic_part = self.C11 - self.C12 - 2 * self.C44
        return (
            self.C12 * np.einsum("ij,kl->ijkl", identity, identity)
            + self.C44 * np.einsum("ik,jl->ijkl", identity, identity)
            + self.C44 * np.einsum("il,jk->ijkl", identity, identity)
            + cubic_part * np.einsum("ia,ja,ka,la->ijkl", axes, axes, axes, axes)
        )
