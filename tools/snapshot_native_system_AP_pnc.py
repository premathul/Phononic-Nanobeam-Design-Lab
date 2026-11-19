"""Apply the original left-incident scattering equations from saved native arrays.

Author: Athul Prem | _AP_pnc
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tau

import numpy as np
from scipy.linalg import solve
from scipy.sparse import issparse, spmatrix


@dataclass(frozen=True)
class VolumeOperators:
    """Shared real elastic stiffness and mass in original body DOF order."""

    stiffness: spmatrix
    mass: spmatrix

    def __post_init__(self):
        if (not issparse(self.stiffness) or not issparse(self.mass)
                or self.stiffness.shape != self.mass.shape
                or self.stiffness.shape[0] != self.stiffness.shape[1]
                or not all(np.issubdtype(matrix.dtype, np.floating) for matrix in (self.stiffness, self.mass))):
            raise ValueError("The native body requires real square sparse K and M of equal size.")

    @property
    def size(self):
        return self.stiffness.shape[0]

    def dynamic_action(self, omega_squared, fields):
        """Apply K − ω²M using real sparse products and only dense field temporaries."""
        action = self.stiffness @ fields.real - omega_squared * (self.mass @ fields.real)
        if np.iscomplexobj(fields):
            action = action + 1j * (self.stiffness @ fields.imag - omega_squared * (self.mass @ fields.imag))
        return action


def _indices(data, name, size):
    indices = np.asarray(data[name])
    if (indices.ndim != 1 or not indices.size or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= size) or np.unique(indices).size != indices.size):
        raise ValueError(f"Invalid native indices in {name}.")
    return indices


def _mode_rows(roots, complete_roots):
    if not np.isin(roots, complete_roots).all():
        raise ValueError("Propagating roots must belong to their complete outgoing basis.")
    return (roots[:, None] == complete_roots).argmax(axis=1)


@dataclass(frozen=True)
class NativeScatteringSystem:
    """Matrix action, physical forcing and port coefficients at one saved frequency.

    The field contains total displacement for the native left-incident drives.
    Each outgoing basis includes its native evanescent modes. Both faces use
    the archived reference planes and the original time convention exp(−iωt).
    """

    volume: VolumeOperators
    frequency_hz: float
    left_dofs: np.ndarray
    right_dofs: np.ndarray
    incident_roots: np.ndarray
    reflection_roots: np.ndarray
    transmission_roots: np.ndarray
    incident_powers_w: np.ndarray
    reflection_powers_w: np.ndarray
    transmission_powers_w: np.ndarray
    _incident_displacement: np.ndarray
    _incident_force: np.ndarray
    _left_basis: np.ndarray
    _right_basis: np.ndarray
    _left_radiation: np.ndarray
    _right_radiation: np.ndarray
    _reflection_rows: np.ndarray
    _transmission_rows: np.ndarray

    @classmethod
    def from_saved_fields(cls, volume, frequency_hz, data):
        """Use archived SAFE arrays in selected_fields_AP_pnc.npz order."""
        omega = tau * frequency_hz
        if frequency_hz <= 0 or not isfinite(omega * omega):
            raise ValueError("The native frequency and its squared angular frequency must be positive and finite.")
        left_dofs = _indices(data, "port_left_dofs", volume.size)
        right_dofs = _indices(data, "port_right_dofs", volume.size)
        count = left_dofs.size
        if right_dofs.size != count or np.intersect1d(left_dofs, right_dofs).size:
            raise ValueError("The two native face maps must be disjoint and have equal size.")
        displacement, axial_force = (np.asarray(data[key]) for key in ("port_displacements", "port_axial_forces"))
        if any(array.shape != (count, 2 * count) or not np.isfinite(array).all()
               for array in (displacement, axial_force)):
            raise ValueError("Expected a complete pair of finite native port bases.")
        left_roots = _indices(data, "reflection_all_mode_indices", 2 * count)
        right_roots = _indices(data, "transmission_all_mode_indices", 2 * count)
        if left_roots.size != count or right_roots.size != count or np.intersect1d(left_roots, right_roots).size:
            raise ValueError("The outgoing roots must partition the complete native port basis.")
        incident, reflection, transmission = (
            _indices(data, f"{kind}_mode_indices", 2 * count) for kind in ("incident", "reflection", "transmission")
        )
        if (not np.array_equal(incident, transmission)
                or not np.array_equal(data["frequencies_hz"], np.full(incident.size, frequency_hz))):
            raise ValueError("Expected the original left-incident drives at the stated native frequency.")
        reflection_rows = _mode_rows(reflection, left_roots)
        transmission_rows = _mode_rows(transmission, right_roots)

        # Tractions in the saved mode arrays use the +x normal on both faces.
        # Solve Z U = F for outward traction with the appropriate outward sign.
        left_basis, right_basis = displacement[:, left_roots], displacement[:, right_roots]
        left_radiation = solve(left_basis.T, -axial_force[:, left_roots].T).T
        right_radiation = solve(right_basis.T, axial_force[:, right_roots].T).T
        incident_displacement = displacement[:, incident]
        incident_force = -axial_force[:, incident]
        signed_power = (omega / 2) * np.sum(displacement.conj() * axial_force, axis=0).imag
        incident_power = signed_power[incident]
        reflection_power = -signed_power[reflection]
        transmission_power = signed_power[transmission]
        if (not all(np.allclose(power, 1.0, atol=1e-8, rtol=0)
                    for power in (incident_power, reflection_power, transmission_power))
                or not np.allclose(incident_power, data["incident_powers_w"], atol=1e-8, rtol=0)):
            raise ValueError("The native propagating roots must carry unit power in their expected directions.")
        return cls(volume, frequency_hz, left_dofs, right_dofs, incident, reflection, transmission,
                   incident_power, reflection_power, transmission_power, incident_displacement, incident_force,
                   left_basis, right_basis, left_radiation, right_radiation, reflection_rows, transmission_rows)

    def apply(self, fields):
        """Apply A(ω) = K − ω²M − Z to body vectors, including both face blocks."""
        fields = np.asarray(fields, dtype=np.complex128)
        if fields.ndim not in (1, 2) or fields.shape[0] != self.volume.size:
            raise ValueError("The first field dimension must be the native volume DOF count.")
        omega = tau * self.frequency_hz
        result = self.volume.dynamic_action(omega * omega, fields)
        result[self.left_dofs] -= self._left_radiation @ fields[self.left_dofs]
        result[self.right_dofs] -= self._right_radiation @ fields[self.right_dofs]
        return result

    def forcing(self):
        """Return B = F_inc − Z_left U_inc in the native incident column order."""
        result = np.zeros((self.volume.size, self.incident_roots.size), dtype=np.complex128)
        result[self.left_dofs] = self._incident_force - self._left_radiation @ self._incident_displacement
        return result

    def outgoing(self, fields):
        """Return native propagating (R, T) amplitudes at the original end planes."""
        field = np.asarray(fields)
        if field.shape != (self.volume.size, self.incident_roots.size):
            raise ValueError("Expected one full body field for each native incident root.")
        reflected = solve(self._left_basis, field[self.left_dofs] - self._incident_displacement)
        transmitted = solve(self._right_basis, field[self.right_dofs])
        return reflected[self._reflection_rows], transmitted[self._transmission_rows]

    def modal_power_fractions(self, reflection, transmission):
        """Return per-outgoing-mode R/T power fractions for native raw amplitudes."""
        blocks = []
        for amplitudes, powers in ((reflection, self.reflection_powers_w), (transmission, self.transmission_powers_w)):
            amplitudes = np.asarray(amplitudes)
            if amplitudes.shape != (powers.size, self.incident_roots.size):
                raise ValueError("Expected outgoing mode by incident root amplitude blocks.")
            blocks.append(abs(amplitudes)**2 * powers[:, None] / self.incident_powers_w[None, :])
        return tuple(blocks)

    def relative_residuals(self, fields, rhs=None):
        """Return each original equation's Euclidean residual relative to its forcing."""
        rhs = self.forcing() if rhs is None else np.asarray(rhs)
        if rhs.shape != (self.volume.size, self.incident_roots.size) or np.shape(fields) != rhs.shape:
            raise ValueError("Expected volume field and forcing columns in the native incident order.")
        rhs_norm = np.linalg.norm(rhs, axis=0)
        if np.any(rhs_norm <= 0) or not np.isfinite(rhs_norm).all():
            raise ValueError("Every native incident forcing must have positive finite norm.")
        return np.linalg.norm(self.apply(fields) - rhs, axis=0) / rhs_norm
