"""Finite-response qualification, separate from completion. Athul Prem | _AP_pnc."""

from __future__ import annotations

import numpy as np


POWER_LIMIT = 0.01
COMPLEX_LIMIT = 0.01
FLUX_GRAM_LIMIT_W = 1e-7


def qualify_scattering(result, modes, *, length_m, uniform, power_balance_passed,
                       matched_reference_passed):
    """Diagnose the saved propagating block without changing any coefficients.

    The coefficient norm is meaningful only after checking the signed flux
    Gram against a unit-power basis. This tests arbitrary combinations of the
    computed incident columns, not the full two-sided scattering matrix.
    Uniform-guide phases use the same discrete SAFE roots and physical planes;
    this is a finite reference check, not a continuum phase-error estimate.
    """
    sign = 1 if result.incident_side == "left" else -1
    groups = {"incident": result.incident_mode_indices,
              "reflection": result.reflection_mode_indices,
              "transmission": result.transmission_mode_indices}
    gram_errors = {}
    for name, indices in groups.items():
        direction = -sign if name == "reflection" else sign
        gram = direction * modes.flux_gram(indices)
        gram_errors[name] = float(np.max(abs(gram - np.eye(len(indices)))))
    combined = np.concatenate((groups["incident"], groups["reflection"]))
    cross = modes.flux_gram(combined)[:len(groups["incident"]), len(groups["incident"]):]
    gram_errors["cross_direction"] = float(np.max(abs(cross)))
    unit_error = float(np.max(abs(result.incident_powers_w - 1)))
    basis_passed = bool(unit_error <= FLUX_GRAM_LIMIT_W and
                        all(np.isfinite(v) and v <= FLUX_GRAM_LIMIT_W for v in gram_errors.values()))
    reflection, transmission = result.reflection_amplitudes, result.transmission_amplitudes
    defect = reflection.conj().T @ reflection + transmission.conj().T @ transmission
    defect -= np.eye(len(groups["incident"]))
    eigenvalues = np.linalg.eigvalsh(defect)
    spectral_norm = float(np.max(abs(eigenvalues)))
    coherent_passed = bool(basis_passed and np.isfinite(spectral_norm) and spectral_norm < POWER_LIMIT)
    coherent = {"status": "passed" if coherent_passed else "failed",
                "scope": f"{result.incident_side.title()}-incident propagating block; not full two-port unitarity",
                "flux_gram_entry_errors_w": gram_errors,
                "unit_incident_power_error_w": unit_error,
                "flux_gram_limit_w": FLUX_GRAM_LIMIT_W,
                "unit_flux_basis_passed": basis_passed,
                "coefficient_defect_spectral_norm": spectral_norm,
                "power_limit": POWER_LIMIT}
    reference = {"status": "not_applicable", "scope": "Uniform guides only"}
    eligible_reference = None
    if uniform:
        distance = sign * length_m
        expected = np.zeros_like(transmission)
        for column, root in enumerate(groups["incident"]):
            rows = np.flatnonzero(groups["transmission"] == root)
            if len(rows) != 1:
                raise ValueError("Uniform-reference transmission roots must match each incident root exactly.")
            expected[rows[0], column] = np.exp(1j * modes.wavenumbers_rad_m[root] * distance)
        complex_error = float(max(np.max(abs(reflection)), np.max(abs(transmission - expected))))
        phase_passed = bool(np.isfinite(complex_error) and complex_error < COMPLEX_LIMIT)
        eligible_reference = bool(power_balance_passed and coherent_passed
                                  and matched_reference_passed is True and phase_passed)
        reference = {"status": "passed" if eligible_reference else "failed",
                     "matched_power_passed": bool(matched_reference_passed),
                     "same_basis_complex_passed": phase_passed,
                     "maximum_complex_error": complex_error, "complex_limit": COMPLEX_LIMIT,
                     "signed_plane_distance_m": distance,
                     "scope": "Original-plane R=0 and T=exp(i*k*distance) in this discrete SAFE basis; no phase subtraction"}
    passed = bool(power_balance_passed and coherent_passed and eligible_reference is not False)
    return {"schema_version": 1, "status": "finite_checks_passed" if passed else "finite_checks_failed",
            "equations_and_column_flux_passed": bool(power_balance_passed),
            "coherent_power": coherent, "uniform_reference": reference,
            "eligible_as_uniform_reference": eligible_reference,
            "spatial_convergence": "unestablished", "frequency_convergence": "unestablished",
            "experimental_validation": "not_established"}
