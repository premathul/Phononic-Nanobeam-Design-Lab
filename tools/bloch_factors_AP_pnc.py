"""Fixed-frequency Bloch factors for a small elastic cell. Athul Prem | _AP_pnc."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import LinAlgError, eig
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import splu

from phononic.elasticity_AP_pnc import ElasticSystem
from phononic.lead_modes_AP_pnc import _end_dof_pairs


@dataclass(frozen=True)
class BlochFactorMode:
    """A near-unit root with a complex, three-component displacement field.

    The displacement is in original volume-basis order in metres. Its left
    trace has Euclidean norm 1 m and a positive real largest component.
    The matched trace ordering is defined by the result's face DOF arrays.
    Power uses this amplitude, with positive sign for flow toward +x.
    """

    root_index: int
    multiplier: complex
    phase_rad: float
    wave_number_rad_m: float
    direction: str
    power_w: float
    relative_power: float
    radial_error: float
    chordal_uncertainty: float
    interior_residual: float
    traction_residual: float
    left_trace: np.ndarray
    right_trace: np.ndarray
    displacement: np.ndarray


@dataclass(frozen=True)
class BlochFactorResult:
    """Bloch factors of the finite discretization at one positive frequency.

    Each homogeneous root column is [alpha, beta] for lambda = alpha/beta.
    Zero beta represents infinity unless alpha is also zero. The condensed
    dynamic stiffness is in N/m and ordered by left_dofs then right_dofs.
    """

    frequency_hz: float
    period_m: float
    left_dofs: np.ndarray
    right_dofs: np.ndarray
    homogeneous_roots: np.ndarray
    boundary_dynamic_stiffness: np.ndarray | None
    modes: tuple[BlochFactorMode, ...]
    status: str
    diagnostics: dict

    @property
    def face_dof_count(self) -> int:
        return len(self.left_dofs)


def _condense(dynamic, boundary, interior):
    """Build the real Schur complement using bounded batches of face loads."""
    schur = dynamic[boundary][:, boundary].toarray()
    if not interior.size:
        return schur, None, None, None

    interior_dynamic = dynamic[interior][:, interior].tocsc()
    factor = splu(interior_dynamic)
    pivot_ratio = float(np.min(abs(factor.U.diagonal())) / np.max(abs(interior_dynamic.data)))
    coupling = dynamic[interior][:, boundary].tocsc()
    reaction = dynamic[boundary][:, interior]
    for start in range(0, len(boundary), 16):
        stop = min(start + 16, len(boundary))
        schur[:, start:stop] -= reaction @ factor.solve(coupling[:, start:stop].toarray())
    return schur, factor, coupling, pivot_ratio


def _root_pairing_error(alpha, beta, target_alpha, target_beta) -> float:
    """Measure a one-to-one pairing by chordal distance, including 0 and infinity."""
    distances = abs(target_alpha[:, None] * beta - target_beta[:, None] * alpha)
    rows, columns = linear_sum_assignment(distances)
    return float(np.max(distances[rows, columns]))


def _companion(schur, face_dofs):
    scale = np.max(abs(schur))
    left = slice(None, face_dofs)
    right = slice(face_dofs, None)
    a0 = schur[right, left] / scale
    a1 = schur[left, left] / scale + schur[right, right] / scale
    a2 = schur[left, right] / scale
    zero = np.zeros_like(a0)
    identity = np.eye(face_dofs)
    return np.block([[zero, identity], [-a0, -a1]]), np.block([[identity, zero], [zero, a2]])


def _ratio(residual, reference) -> float:
    if reference == 0.0:
        return 0.0 if residual == 0.0 else math.inf
    return float(residual / reference)


def solve_bloch_factors(
    system: ElasticSystem,
    frequency_hz: float,
    *,
    max_face_dofs: int = 512,
    max_body_dofs: int = 30_000,
    unit_circle_tolerance: float = 1e-8,
    residual_tolerance: float = 1e-8,
    relative_power_tolerance: float = 1e-8,
    reciprocity_tolerance: float = 1e-7,
) -> BlochFactorResult:
    """Solve a bounded 3D cell at real frequency with exp(-i omega t).

    Natural side boundaries are free. With paired face traces u_R = lambda u_L
    and outward reaction vectors F_L and F_R, equilibrium requires
    lambda F_L + F_R = 0. Eliminating the interior gives the quadratic
    S_LR lambda^2 + (S_LL + S_RR) lambda + S_RL in the left trace.

    Near-unit factors retain the computed modulus and multiplier phase.
    The phase uses the principal zone. Propagation direction is set by
    signed energy flux. The dimensionless cutoff gate compares absolute
    flux to omega times the cell energy omega^2 u^H M u / 2. A near-unit
    mode with unresolved modulus, periodic equilibrium or flux is retained
    with direction 'unresolved'. Status describes the near-unit roots and
    the homogeneous spectrum's pairing checks for this discretization.
    """
    if (not isinstance(frequency_hz, Real) or isinstance(frequency_hz, (bool, np.bool_))
            or not math.isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError("Bloch frequency must be positive and finite.")
    omega = math.tau * float(frequency_hz)
    omega_squared = omega * omega
    if not math.isfinite(omega_squared) or omega_squared <= 0:
        raise ValueError("Squared angular frequency must be positive and finite.")
    for name, value in (("Face", max_face_dofs), ("Body", max_body_dofs)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} DOF cap must be a positive integer.")
    tolerances = (
        unit_circle_tolerance, residual_tolerance, relative_power_tolerance, reciprocity_tolerance,
    )
    if any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
           or not math.isfinite(value) or not 0 < value < 1 for value in tolerances):
        raise ValueError("Bloch tolerances must be positive, finite and smaller than one.")

    size = system.basis.N
    if size > max_body_dofs:
        raise ValueError(f"Cell has {size} body DOFs, exceeding cap {max_body_dofs}.")
    period, left, right = _end_dof_pairs(system)
    face_count = len(left)
    if face_count > max_face_dofs:
        raise ValueError(f"Cell has {face_count} DOFs per face, exceeding cap {max_face_dofs}.")
    boundary = np.concatenate((left, right))
    interior = np.setdiff1d(np.arange(size), boundary)
    dynamic = system.K - omega_squared * system.M
    if np.iscomplexobj(dynamic.data) or not np.isfinite(dynamic.data).all():
        raise ValueError("Cell dynamic stiffness must be real and finite.")
    dynamic = dynamic.astype(float, copy=False)

    eps = np.finfo(float).eps
    roots = np.empty((2, 0), dtype=complex)
    schur = None
    modes = []
    issues = []
    diagnostics = {
        "interior_dof_count": int(len(interior)),
        "unit_circle_tolerance": unit_circle_tolerance,
        "residual_tolerance": residual_tolerance,
        "relative_power_tolerance": relative_power_tolerance,
        "reciprocity_tolerance": reciprocity_tolerance,
    }

    def finish():
        diagnostics["issues"] = issues
        diagnostics["right_count"] = sum(mode.direction == "right" for mode in modes)
        diagnostics["left_count"] = sum(mode.direction == "left" for mode in modes)
        diagnostics["unresolved_count"] = sum(mode.direction == "unresolved" for mode in modes)
        return BlochFactorResult(
            float(frequency_hz), period, left, right, roots, schur, tuple(modes),
            "unresolved" if issues else "resolved", diagnostics,
        )

    try:
        schur, factor, coupling, pivot_ratio = _condense(dynamic, boundary, interior)
    except RuntimeError as exc:
        issues.append(f"Interior dynamic stiffness factorization failed: {exc}")
        return finish()
    diagnostics["interior_minimum_pivot_ratio"] = pivot_ratio
    if pivot_ratio is not None and pivot_ratio <= 128 * eps:
        issues.append("Interior dynamic stiffness is unresolved near a fixed-face resonance.")
        schur = None
        return finish()
    if not np.isfinite(schur).all() or not np.any(schur):
        issues.append("Condensed dynamic stiffness is nonfinite or zero.")
        schur = None
        return finish()
    symmetry_error = float(np.linalg.norm(schur - schur.T, np.inf) / np.linalg.norm(schur, np.inf))
    diagnostics["schur_symmetry_error"] = symmetry_error
    if symmetry_error > residual_tolerance:
        issues.append("The condensed dynamic stiffness failed its symmetry check.")

    first, second = _companion(schur, face_count)
    try:
        roots, dual_vectors, vectors = eig(first, second, homogeneous_eigvals=True, left=True)
    except LinAlgError as exc:
        issues.append(f"Generalized Bloch eigenproblem failed: {exc}")
        return finish()

    root_scale = np.max(abs(roots), axis=0)
    regular = np.isfinite(roots).all(axis=0) & (root_scale > 0)
    diagnostics["indeterminate_root_count"] = int(np.count_nonzero(~regular))
    diagnostics["infinite_root_count"] = int(np.count_nonzero(regular & (roots[1] == 0)))
    diagnostics["zero_root_count"] = int(np.count_nonzero(regular & (roots[0] == 0)))
    if not np.all(regular):
        issues.append("The Bloch pencil returned indeterminate homogeneous roots.")
    regular_indices = np.flatnonzero(regular)
    coordinates = roots[:, regular] / root_scale[regular]
    alpha, beta = coordinates / np.linalg.norm(coordinates, axis=0)
    if alpha.size:
        conjugacy_error = _root_pairing_error(alpha, beta, alpha.conj(), beta.conj())
        reciprocal_error = _root_pairing_error(alpha, beta, beta, alpha)
        diagnostics["conjugacy_chordal_error"] = conjugacy_error
        diagnostics["reciprocity_chordal_error"] = reciprocal_error
        if max(conjugacy_error, reciprocal_error) > reciprocity_tolerance:
            issues.append("The homogeneous roots failed the conjugate or reciprocal pairing check.")

    # The derivative along a unit tangent to [alpha, beta] in projective
    # coordinates is proportional to alpha* A + beta* B. The dual overlap
    # estimates roundoff sensitivity without dividing by a small beta.
    active_vectors = vectors[:, regular]
    tangent = (first @ active_vectors) * alpha.conj() + (second @ active_vectors) * beta.conj()
    dual_overlap = abs(np.sum(dual_vectors[:, regular].conj() * tangent, axis=0))
    pencil_scale = np.linalg.norm(first, np.inf) + np.linalg.norm(second, np.inf)
    uncertainty = np.full(alpha.size, math.inf)
    np.divide(32 * eps * pencil_scale, dual_overlap, out=uncertainty, where=dual_overlap > 0)
    circle_distance = abs(abs(alpha) - abs(beta)) / math.sqrt(2)
    # A distance of roughly half the requested radial tolerance corresponds
    # to a unit-circle multiplier. Retain nearby and poorly conditioned roots
    # for diagnostics even when their actual modulus fails the specified gate.
    candidates = (alpha != 0) & (beta != 0) & (
        circle_distance <= np.maximum(4 * unit_circle_tolerance, 8 * uncertainty)
    )
    candidate_indices = np.flatnonzero(candidates)
    diagnostics["near_unit_root_count"] = int(len(candidate_indices))
    for local_index in candidate_indices:
        root_index = int(regular_indices[local_index])
        multiplier = complex(roots[0, root_index] / roots[1, root_index])
        if not math.isfinite(abs(multiplier)):
            issues.append(f"Root {root_index} has an unresolved finite multiplier.")
            continue

        trace = vectors[:face_count, root_index].astype(complex)
        pivot = trace[np.argmax(abs(trace))]
        if pivot == 0.0:
            issues.append(f"Root {root_index} has a zero displacement trace.")
            continue
        trace /= pivot
        trace /= np.linalg.norm(trace)
        displacement = np.empty(size, dtype=complex)
        displacement[left] = trace
        displacement[right] = multiplier * trace
        if factor is not None:
            load = -(coupling @ displacement[boundary])
            response = factor.solve(np.column_stack((load.real, load.imag)))
            displacement[interior] = response[:, 0] + 1j * response[:, 1]
        stiffness_force = system.K @ displacement
        inertial_force = omega_squared * (system.M @ displacement)
        reaction = stiffness_force - inertial_force
        force_left, force_right = reaction[left], reaction[right]
        interior_residual = _ratio(
            np.linalg.norm(reaction[interior]),
            np.linalg.norm(stiffness_force[interior]) + np.linalg.norm(inertial_force[interior]),
        )
        traction_residual = _ratio(
            np.linalg.norm(multiplier * force_left + force_right),
            abs(multiplier) * np.linalg.norm(force_left) + np.linalg.norm(force_right),
        )
        work = np.vdot(displacement[right], force_right)
        power = float(omega / 2 * work.imag)
        energy = float(np.vdot(displacement, inertial_force).real / 2)
        relative_power = _ratio(abs(power), omega * energy) if energy > 0 else math.nan
        relative_work = _ratio(abs(work.imag), np.linalg.norm(displacement[right]) * np.linalg.norm(force_right))

        radial_error = abs(abs(multiplier) - 1)
        chordal_uncertainty = float(uncertainty[local_index])
        other_candidates = candidate_indices[candidate_indices != local_index]
        separation = abs(alpha[local_index] * beta[other_candidates] - beta[local_index] * alpha[other_candidates])
        resolution = 8 * (uncertainty[other_candidates] + chordal_uncertainty)
        mode_issues = []
        if radial_error > unit_circle_tolerance or 8 * chordal_uncertainty > unit_circle_tolerance / 2:
            mode_issues.append("unit modulus")
        if np.any(separation <= resolution):
            mode_issues.append("a distinct trace at a coincident root")
        if (not math.isfinite(interior_residual) or not math.isfinite(traction_residual)
                or max(interior_residual, traction_residual) > residual_tolerance):
            mode_issues.append("original equations")
        if (not math.isfinite(relative_power) or relative_power <= relative_power_tolerance
                or not math.isfinite(relative_work) or relative_work <= 128 * eps):
            mode_issues.append("signed flux")
        direction = "right" if power > 0 else "left"
        if mode_issues:
            direction = "unresolved"
            issues.append(f"Root {root_index} could not resolve {', '.join(mode_issues)}.")
        phase = float(np.angle(multiplier))
        modes.append(BlochFactorMode(
            root_index, multiplier, phase, phase / period, direction, power, relative_power,
            radial_error, chordal_uncertainty, interior_residual, traction_residual,
            displacement[left], displacement[right], displacement,
        ))

    if sum(mode.direction == "right" for mode in modes) != sum(mode.direction == "left" for mode in modes):
        issues.append("The resolved propagating counts differ between the two directions.")
    modes.sort(key=lambda mode: mode.phase_rad)
    return finish()
