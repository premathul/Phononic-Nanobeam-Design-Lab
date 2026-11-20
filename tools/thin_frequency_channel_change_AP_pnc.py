"""Reconcile temporal channel power changes on fixed meshes. Athul Prem | _AP_pnc."""

from __future__ import annotations

import numpy as np

from tools.compare_scattering_AP_pnc import GROUPS, PHASE_AMPLITUDE_FLOOR, Modes, align, require

ANCHOR = (0, 2)
FOCUSED_DRIVES = (3, 5)
POWER_ROUNDOFF = 1e-12
RECIPROCAL_WAVENUMBER_RTOL = 1e-12


def _outgoing_orders(modes, incoming, section_mass):
    modes_in = modes["incoming"]
    roots, k = modes_in.roots[incoming], modes_in.k[incoming]
    plus = roots[:, None] == modes["transmission"].roots
    minus = np.isclose(k[:, None], -modes["reflection"].k[None, :], atol=0, rtol=RECIPROCAL_WAVENUMBER_RTOL)
    require(all(np.all(mask.sum(axis=0) == 1) and np.all(mask.sum(axis=1) == 1) for mask in (plus, minus)),
            "The native through roots or opposite-wavenumber return partners are ambiguous.")
    reversed_trace = Modes(roots, -k, modes_in.displacement[..., incoming].conj())
    partners, _, match = align(reversed_trace, modes["reflection"], section_mass, None, False)
    require(np.array_equal(partners, np.argmax(minus, axis=1)),
            "Conjugate return traces and opposite-wavenumber partners disagree.")
    return {"incoming": incoming, "reflection": partners, "transmission": np.argmax(plus, axis=1)}, min(match["mass_overlaps"])


def _point_in_anchor_order(point, match_order, anchor_order, section_mass):
    indices = {}
    size = len(anchor_order["incoming"])
    for group in GROUPS:
        order = np.asarray(match_order[group])
        require(order.shape == (size,) and order.dtype.kind in "iu" and np.array_equal(np.sort(order), np.arange(size)),
                f"Invalid anchor-to-native {group} channel permutation.")
        indices[group] = order[anchor_order[group]]
    local_partners, conjugate_overlap = _outgoing_orders(point.modes, indices["incoming"], section_mass)
    require(all(np.array_equal(indices[group], local_partners[group]) for group in GROUPS[1:]),
            "The frequency-local match does not preserve through and reciprocal return identities.")
    rows = point.record["native_drives"]
    require([row["native_root_id"] for row in rows] == point.modes["incoming"].roots.tolist(),
            "Saved native power rows use different incident roots.")

    blocks, totals = {}, {}
    for group in GROUPS[1:]:
        raw, powers = np.asarray(point.blocks[group]), np.asarray(point.modal_powers[group])
        require(raw.shape == powers.shape == (size, size) and np.isfinite(raw).all() and np.isfinite(powers).all()
                and np.all(powers >= 0) and np.all((abs(raw) != 0) | (powers == 0)),
                f"Invalid native complex block or flux-normalized modal power for {group}.")
        magnitude = abs(raw)
        phasor = np.divide(raw, magnitude, out=np.zeros(raw.shape, dtype=complex), where=magnitude > 0)
        order = np.ix_(indices[group], indices["incoming"])
        blocks[group] = (phasor * np.sqrt(powers))[order]
        totals[group] = np.array([rows[i][group] for i in indices["incoming"]])
        require(np.isfinite(totals[group]).all() and np.all(totals[group] >= 0)
                and np.max(abs(powers[order].sum(axis=0) - totals[group])) <= POWER_ROUNDOFF,
                f"The normalized {group} block does not reproduce the saved native totals.")
    return indices, blocks, totals, conjugate_overlap


def _difference(lower, upper, left, right, matching):
    left_order, left_blocks, left_power, _ = left
    right_order, right_blocks, right_power, _ = right
    phases = {}
    for group in GROUPS:
        match = matching[group]
        order = np.asarray(match["candidate_order"])
        angle = np.asarray(match["candidate_gauge_phase_rad"])
        left_roots, right_roots = lower.modes[group].roots, upper.modes[group].roots
        require(match["status"] == "matched" and order.shape == angle.shape == left_roots.shape
                and order.dtype.kind in "iu" and np.array_equal(np.sort(order), np.arange(len(left_roots)))
                and np.isfinite(angle).all() and np.array_equal(order[left_order[group]], right_order[group])
                and np.array_equal(match["reference_roots"], left_roots)
                and np.array_equal(match["candidate_roots"], right_roots[order]),
                f"The edge {group} phase match differs from the verified channel order.")
        phases[group] = np.exp(1j * angle[left_order[group]])

    aligned, changes, native_change, errors = {}, {}, {}, {}
    for group in GROUPS[1:]:
        before = left_blocks[group]
        after = phases[group][:, None] * right_blocks[group] / phases["incoming"]
        aligned[group] = after
        changes[group] = ((after + before).conj() * (after - before)).real
        native_change[group] = right_power[group] - left_power[group]
        errors[group] = float(np.max(abs(changes[group].sum(axis=0) - native_change[group])))
        direct = abs(right_blocks[group])**2 - abs(before)**2
        require(max(errors[group], float(np.max(abs(changes[group] - direct)))) <= POWER_ROUNDOFF,
                f"The analytic {group} channel changes do not close to the native per-drive power change.")
    return aligned, changes, native_change, errors


def frequency_local_block_changes(points, matches, channel_orders, *, section_mass):
    """Split adjacent same-mesh T/R differences in the 225 nm, 1 GHz labels.

    Points supply native_modes, amplitudes and modal powers via ``modes``,
    ``blocks`` and ``modal_powers``, plus the saved native-drive ``record``.
    Signed matches, anchor-to-native channel orders and the shared section
    mass come from the main replay. Entry power changes use the real part of
    ``conj(b+a) * (b-a)`` after phase alignment, with exact native flux weights
    and the same physical reference planes. Reflected conversions exclude
    the unique opposite-wavenumber conjugate trace of each incident channel.
    """
    size = 5
    anchor_order, _ = _outgoing_orders(points[ANCHOR].modes, np.arange(size), section_mass)
    endpoints = {key: _point_in_anchor_order(point, channel_orders[key], anchor_order, section_mass)
                 for key, point in points.items()}
    edges = []
    focus = np.array(FOCUSED_DRIVES) - 1
    for lower_key in sorted(points):
        mesh, frequency = lower_key
        upper_key = (mesh, frequency + 1)
        if upper_key not in points:
            continue
        lower, upper = points[lower_key], points[upper_key]
        require(all(lower.record[key] == upper.record[key] for key in ("mesh_hash", "reference_planes_x_m")),
                "Temporal channel changes require the identical mesh and original reference planes.")
        left, right = endpoints[lower_key], endpoints[upper_key]
        aligned, changes, native, errors = _difference(lower, upper, left, right, matches[lower_key, upper_key])
        transmission = changes["transmission"]
        selected = np.zeros_like(transmission)
        selected[np.ix_(focus, focus)] = transmission[np.ix_(focus, focus)]
        remaining = transmission - selected
        reflection = changes["reflection"]
        conversion = np.where(np.eye(size, dtype=bool), 0.0, reflection)
        focus_terms = []
        for incident in focus:
            for outgoing in focus:
                before = left[1]["transmission"][outgoing, incident]
                after = aligned["transmission"][outgoing, incident]
                focus_terms.append({"incident_drive": int(incident + 1), "outgoing_drive": int(outgoing + 1),
                                    "lower_flux_amplitude_re_im": [float(before.real), float(before.imag)],
                                    "upper_aligned_flux_amplitude_re_im": [float(after.real), float(after.imag)],
                                    "wrapped_phase_change_degrees": float(np.angle(after / before, deg=True))
                                    if min(abs(before), abs(after)) > PHASE_AMPLITUDE_FLOOR else None,
                                    "delta_T_percentage_points": float(100 * transmission[outgoing, incident])})
        t_focus, t_remaining = selected.sum(axis=0), remaining.sum(axis=0)
        r_same, r_conversion = reflection.diagonal(), conversion.sum(axis=0)
        drives = [{"drive": i + 1, "delta_T_percentage_points": float(100 * native["transmission"][i]),
                   "focused_delta_T_percentage_points": float(100 * t_focus[i]),
                   "remaining_delta_T_percentage_points": float(100 * t_remaining[i]),
                   "delta_R_percentage_points": float(100 * native["reflection"][i]),
                   "same_channel_delta_R_percentage_points": float(100 * r_same[i]),
                   "converted_delta_R_percentage_points": float(100 * r_conversion[i])} for i in range(size)]
        edges.append({"lower_point_indices": list(lower_key), "upper_point_indices": list(upper_key),
                      "root_ids_by_anchor_drive_at_endpoints": [
                          {group: point.modes[group].roots[prepared[0][group]].tolist() for group in GROUPS}
                          for point, prepared in ((lower, left), (upper, right))],
                      "minimum_reciprocal_conjugate_mass_overlap_at_endpoints": [left[3], right[3]],
                      "focus_transmission_terms": focus_terms, "drives": drives,
                      "reflection_conversion_delta_percentage_points_by_outgoing_and_incident": (100 * conversion).tolist(),
                      "maximum_native_power_closure_fraction": errors})
    return {"anchor_point_indices": list(ANCHOR), "focused_drives": list(FOCUSED_DRIVES),
            "power_roundoff_fraction": POWER_ROUNDOFF, "edges": edges}
