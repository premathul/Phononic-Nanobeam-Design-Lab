"""3D lossless elastic scattering with matched waveguide ports.

Author: Athul Prem | _AP_pnc
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tau
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import solve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .elasticity_AP_pnc import ElasticSystem

if TYPE_CHECKING:
    from .port_modes_AP_pnc import PortModes


@dataclass(frozen=True)
class ScatteringResult:
    """One frequency with a unit-power incident mode in each result column.

    Mode indices address the supplied PortModes arrays and are zero-based.
    Reflection/transmission amplitude and modal-power rows use the corresponding
    outgoing mode indices. Fields are full 3D displacement DOFs, including both
    end faces. Full amplitude matrices include all outgoing evanescent modes.
    The modal, Gram and direct traction fluxes are ratios to incident power.
    The two balance arrays are 1 − R − T. The net outward power instead uses
    the total field on both faces, including interference with the incident wave.
    Condition numbers are those of the unscaled displacement bases, left then
    right. A large value indicates sensitivity of DtN construction and modal
    decomposition to numerical perturbations.
    """

    frequency_hz: float
    incident_side: str
    incident_mode_indices: np.ndarray
    incident_powers_w: np.ndarray
    reflection_mode_indices: np.ndarray
    transmission_mode_indices: np.ndarray
    reflection_amplitudes: np.ndarray
    transmission_amplitudes: np.ndarray
    reflection_all_mode_indices: np.ndarray
    transmission_all_mode_indices: np.ndarray
    reflection_amplitudes_full: np.ndarray
    transmission_amplitudes_full: np.ndarray
    reflection_modal_powers: np.ndarray
    transmission_modal_powers: np.ndarray
    reflection: np.ndarray
    transmission: np.ndarray
    power_balance: np.ndarray
    reflection_gram: np.ndarray
    transmission_gram: np.ndarray
    reflection_flux: np.ndarray
    transmission_flux: np.ndarray
    flux_power_balance: np.ndarray
    net_outward_power_ratio: np.ndarray
    relative_residuals: np.ndarray
    port_basis_condition: tuple[float, float]
    displacements: np.ndarray


def _column_power(displacements: np.ndarray, forces: np.ndarray, omega: float) -> np.ndarray:
    """Time-averaged power for the normal defining the input force columns."""
    return (omega / 2) * np.sum(displacements.conj() * forces, axis=0).imag


def solve_scattering(
    system: ElasticSystem,
    port_modes: PortModes,
    frequency_hz: float,
    incident_side: str = "left",
) -> ScatteringResult:
    """Solve free-sidewall harmonic scattering for every incident propagating mode.

    The port discretization must be assembled from this elastic system and use
    the same cross-section at both ends. Under exp(-i omega t), axial forces in
    PortModes use the +x normal. The left outward normal reverses those forces.
    All outgoing waves, including decaying waves, enter the end traction map.
    """
    if (not isinstance(frequency_hz, Real) or isinstance(frequency_hz, (bool, np.bool_))
            or not isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError("frequency_hz must be a positive finite real number")
    if frequency_hz != port_modes.frequency_hz:
        raise ValueError("port modes and scattering must use the same frequency")
    if incident_side not in ("left", "right"):
        raise ValueError("incident_side must be left or right")
    port = port_modes.port
    if port.elastic_system is not system:
        raise ValueError("port DOF mapping must come from the scattering elastic system")

    left_dofs, right_dofs = port.left_dofs, port.right_dofs
    left_modes, right_modes = port_modes.negative_indices, port_modes.positive_indices
    displacements = port_modes.displacements
    axial_forces = port_modes.axial_forces
    omega = tau * frequency_hz
    omega_squared = omega * omega
    if not isfinite(omega_squared):
        raise ValueError("squared angular frequency must be finite")

    # Z maps interface displacement to outward nodal force. Solve Z U = F
    # by transposing so the displacement basis is the left solve operand.
    u_left = displacements[:, left_modes]
    u_right = displacements[:, right_modes]
    basis_condition = (float(np.linalg.cond(u_left)), float(np.linalg.cond(u_right)))
    if not all(isfinite(condition) for condition in basis_condition):
        raise ValueError("outgoing waveguide displacement basis is singular")
    z_left = solve(u_left.T, -axial_forces[:, left_modes].T).T
    z_right = solve(u_right.T, axial_forces[:, right_modes].T).T
    all_dofs = np.stack((left_dofs, right_dofs))
    port_dofs = u_left.shape[0]
    traction_map = coo_matrix(
        (np.stack((z_left, z_right)).ravel(),
         (np.repeat(all_dofs, port_dofs, axis=1).ravel(),
          np.tile(all_dofs, (1, port_dofs)).ravel())),
        shape=system.K.shape,
    ).tocsc()

    if incident_side == "left":
        incoming_dofs, through_dofs = left_dofs, right_dofs
        reflection_indices, transmission_indices = left_modes, right_modes
        u_reflect, u_transmit, z_in = u_left, u_right, z_left
        normal_sign = -1
    else:
        incoming_dofs, through_dofs = right_dofs, left_dofs
        reflection_indices, transmission_indices = right_modes, left_modes
        u_reflect, u_transmit, z_in = u_right, u_left, z_right
        normal_sign = 1

    incident_indices = transmission_indices[port_modes.propagating[transmission_indices]]
    if len(incident_indices) == 0:
        raise ValueError(f"there are no propagating incident modes at {frequency_hz:g} Hz")
    incident_u = displacements[:, incident_indices]
    incident_force = normal_sign * axial_forces[:, incident_indices]
    incident_power = -_column_power(incident_u, incident_force, omega)
    if not np.all(np.isfinite(incident_power) & (incident_power > 0)):
        raise ValueError("all incident modes must carry finite inward power")

    dynamic = system.K - omega_squared * system.M
    operator = dynamic.tocsc() - traction_map
    rhs = np.zeros((system.K.shape[0], len(incident_indices)), dtype=operator.dtype)
    rhs[incoming_dofs] = incident_force - z_in @ incident_u
    full_u = splu(operator).solve(rhs)
    if not np.all(np.isfinite(full_u)):
        raise RuntimeError("harmonic scattering solve produced a nonfinite field")

    # Forces recovered from the original finite-domain equation D u = F_outward
    # give a separate flux check that does not use outgoing mode coefficients.
    total_force = dynamic @ full_u
    reflected_u = full_u[incoming_dofs] - incident_u
    transmitted_u = full_u[through_dofs]
    reflected_force = total_force[incoming_dofs] - incident_force
    transmitted_force = total_force[through_dofs]
    reflection_flux = _column_power(reflected_u, reflected_force, omega) / incident_power
    transmission_flux = _column_power(transmitted_u, transmitted_force, omega) / incident_power
    net_outward_flux = (_column_power(full_u[incoming_dofs], total_force[incoming_dofs], omega)
                        + _column_power(transmitted_u, transmitted_force, omega)) / incident_power

    reflect_all = solve(u_reflect, reflected_u)
    transmit_all = solve(u_transmit, transmitted_u)
    reflect_propagating = port_modes.propagating[reflection_indices]
    transmit_propagating = port_modes.propagating[transmission_indices]
    reflect_ids = reflection_indices[reflect_propagating]
    transmit_ids = transmission_indices[transmit_propagating]
    reflect = reflect_all[reflect_propagating]
    transmit = transmit_all[transmit_propagating]
    r_modes = abs(reflect)**2 * abs(port_modes.powers_w[reflect_ids, None]) / incident_power
    t_modes = abs(transmit)**2 * abs(port_modes.powers_w[transmit_ids, None]) / incident_power
    reflection = r_modes.sum(axis=0)
    transmission = t_modes.sum(axis=0)
    r_gram = np.sum(reflect_all.conj() * (port_modes.flux_gram(reflection_indices) @ reflect_all), axis=0).real
    t_gram = np.sum(transmit_all.conj() * (port_modes.flux_gram(transmission_indices) @ transmit_all), axis=0).real

    imbalance = operator @ full_u - rhs
    operator_norm = float(np.asarray(abs(operator).sum(axis=1)).max())
    relative_residuals = np.max(abs(imbalance), axis=0) / (
        operator_norm * np.max(abs(full_u), axis=0) + np.max(abs(rhs), axis=0)
    )
    return ScatteringResult(
        frequency_hz=float(frequency_hz),
        incident_side=incident_side,
        incident_mode_indices=incident_indices,
        incident_powers_w=incident_power,
        reflection_mode_indices=reflect_ids,
        transmission_mode_indices=transmit_ids,
        reflection_amplitudes=reflect,
        transmission_amplitudes=transmit,
        reflection_all_mode_indices=reflection_indices,
        transmission_all_mode_indices=transmission_indices,
        reflection_amplitudes_full=reflect_all,
        transmission_amplitudes_full=transmit_all,
        reflection_modal_powers=r_modes,
        transmission_modal_powers=t_modes,
        reflection=reflection,
        transmission=transmission,
        power_balance=1 - reflection - transmission,
        reflection_gram=normal_sign * r_gram / incident_power,
        transmission_gram=-normal_sign * t_gram / incident_power,
        reflection_flux=reflection_flux,
        transmission_flux=transmission_flux,
        flux_power_balance=1 - reflection_flux - transmission_flux,
        net_outward_power_ratio=net_outward_flux,
        relative_residuals=relative_residuals,
        port_basis_condition=basis_condition,
        displacements=full_u,
    )
