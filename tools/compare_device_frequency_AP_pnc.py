"""Compare the bounded 0.8875/1 GHz saved device points. Athul Prem | _AP_pnc.

The relation phase reads only saved section arrays. It never solves an
eigenproblem, assembles a body, or follows an unobserved frequency branch.
"""
from __future__ import annotations

import argparse
import csv
from html import escape
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from phononic.workflow_AP_pnc import digest
from phononic.report_style_AP_pnc import available_link, report_link, report_page, scroll_table
from tools import compare_scattering_AP_pnc as matching

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
REFERENCE_HZ, CANDIDATE_HZ, PORT_DOFS = 1_000_000_000, 887_500_000, 975
RELATIVE_FLOOR = 1e-12
TOLERANCE = 1e-10
PAIR = ((1877, 1838), (1837, 1878))
ACCEPTED_RELATION_SHA256 = "6792e815654f5cc882310ab51a70f80ff5a0280e7f5db62df2ec1a30feaf8e65"
# This immutable section analysis predates the scoped comparison-consumer checks.
RELATION_PRODUCER_SHA256 = "877405161e5dddc6f2f8840a7f8249ad66bd6971cc75079dad276c341028806d"
REPLAY_SOURCE_SHA256 = "f8867e1d8f92630411c7bbb0ad849df52fb34ad49c974081fd1abf69230e9001"
SHARED_REPLAY_SHA256 = "83c98faae6f07071394fc472a1f44d96897258e09b6266db60402da63e06a6e0"
BASE = "results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc/"
SAFE = BASE + "thickness-h80-safe-1335308/"
PUBLIC = BASE + "thickness-h80-export-1335341/historical_26hole_layered_h80_ny12_z6-585ccac1d4f2/"
INTERVAL = "results/runs/port_count_interval_AP_pnc/study/"
PINS = {
    "safe_receipt": (SAFE + "thickness_h80_safe_AP_pnc.json", "b5c0d73656e90aa5ba312939c0ef25633af5a775974f5fc778031584d55c78f3"),
    "section": (SAFE + "ports_AP_pnc.npz", "a13a58e47dee91e4c7d75bb9ec77feb7656ae674392c2f1d7485129da76d0e6a"),
    "interval_inputs": (INTERVAL + "inputs_AP_pnc.json", "e14c59270fcf53d0b26ec41fe46db25ec0d208936101e1f456c50ecf9b05ed26"),
    "candidate_point": (INTERVAL + "point-887500000_AP_pnc.json", "a53557ae5365570a39046b68f12eb0fd0e9e620d7df7d080561ec998e552e6e8"),
    "candidate_modes": (INTERVAL + "modes-887500000_AP_pnc.npz", "93e4e089e6b329244b204ca10e3612653824b1e569476a51ea1924654a83b81c"),
    "interval_verification": ("verification/port_count_interval_AP_pnc.json", "ada05174f12e937bc709b124de6ff97956381ceceb9f2f8595c863b91bdd517f"),
    "reference_summary": (PUBLIC + "summary_AP_pnc.json", "040a219f54446219b6a610cf08382ff7e93a6dcb01f31f3be810977651301c4e"),
    "reference_totals": (PUBLIC + "scattering_AP_pnc.csv", "54cb1ff9cf3d69cef63f3c9858ec7587efee7cec4fe9324751f7094ea323598a"),
    "reference_coefficients": (PUBLIC + "scattering_channels_AP_pnc.csv", "bca65a396d45a15c0f3b8ba58a5d85e9add2b1a7b24b3ec111bbebb0310d6014"),
    "conversion_verification": ("verification/one_ghz_conversion_AP_pnc.json", "f9a72bdcb173c31b96fc0d6d21e589ba9a3a94aa3c9d57877bfd7d6fd90459fd"),
}
SCOPE = ("Two finite frequencies on the saved h80/z6 device, at original 0/16.65 micrometre end planes. "
         "Frequency differences are physical response variation in this discretized model, not a mesh-convergence "
         "test, numerical uncertainty bound, conversion mechanism or continuous-branch proof. All previous failed "
         "1 GHz spatial comparisons remain failed; spatial and frequency convergence remain unresolved.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def link(path, workspace=ROOT):
    path, workspace = Path(path).resolve(), Path(workspace).resolve()
    return {"path": path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else str(path),
            "sha256": sha(path), "bytes": path.stat().st_size}


def pinned(workspace, role, inputs):
    root = Path(workspace).resolve()
    relative, expected = PINS[role]
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file() and sha(path) == expected,
            f"The accepted identity differs for {role}.")
    inputs[role] = link(path, root)
    return path


def json_read(path):
    return json.loads(Path(path).read_bytes())


def validate_metadata(receipt, point, inputs, ledger):
    require(receipt["status"] == "safe_ready" and receipt["frequency_hz"] == REFERENCE_HZ
            and receipt["trace_dofs"] == PORT_DOFS and receipt["discrete_roots"] == 2 * PORT_DOFS
            and receipt["ports"]["sha256"] == PINS["section"][1], "The archived section receipt differs.")
    require(inputs["identity_sha256"] == digest(inputs["identity"])
            and point["identity_sha256"] == inputs["identity_sha256"]
            and point["point_sha256"] == digest({k: v for k, v in point.items() if k != "point_sha256"})
            and point["frequency_hz"] == CANDIDATE_HZ and point["status"] == "accepted"
            and point["modes"]["sha256"] == PINS["candidate_modes"][1]
            and point["modes"]["path"] == Path(PINS["candidate_modes"][0]).name,
            "The accepted candidate point or its section identity differs.")
    archive = inputs["identity"]["archive"]
    require(archive["safe_receipt"]["sha256"] == PINS["safe_receipt"][1]
            and archive["section_modes"]["sha256"] == PINS["section"][1]
            and archive["trace_sha256"] == receipt["trace_sha256"], "The shared section/trace provenance differs.")
    metrics = point["metrics"]
    require(metrics["positive_open_channels"] == metrics["negative_open_channels"] == 5
            and metrics["discrete_roots"] == 2 * PORT_DOFS
            and metrics["degeneracy_status"] == "none_detected"
            and not metrics["coincident_propagating_root_groups"], "The candidate lacks five resolved signed channels.")
    verified = ledger["independent_verification"]
    accepted = [row for row in verified["points"] if row["frequency_hz"] == CANDIDATE_HZ]
    require(ledger["status"] == "independently_verified_finite_count_interval"
            and verified["status"] == "verified_finite_count_interval"
            and verified["identity_sha256"] == point["identity_sha256"]
            and len(accepted) == 1 and accepted[0]["point_sha256"] == point["point_sha256"],
            "Independent interval replay does not bind this accepted point.")


def read_signed_modes(path, frequency_hz, positive_roots, negative_roots, *, dofs=PORT_DOFS):
    """Keep complete partitions and signed transport; root numbers supply no ordering physics."""
    require(Path(path).stat().st_size < 64 * 2**20, "Saved section archive exceeds the bounded 64 MiB input.")
    with np.load(path, allow_pickle=False) as arrays:
        k, propagating = arrays["wavenumbers_rad_m"], arrays["propagating"]
        powers, residuals = arrays["powers_w"], arrays["relative_residuals"]
        positive, negative = arrays["positive_indices"], arrays["negative_indices"]
        require(float(arrays["frequency_hz"]) == frequency_hz
                and k.shape == propagating.shape == powers.shape == residuals.shape == (2 * dofs,)
                and propagating.dtype == np.bool_ and positive.shape == negative.shape == (dofs,)
                and positive.dtype.kind in "iu" and negative.dtype.kind in "iu"
                and np.array_equal(np.sort(np.r_[positive, negative]), np.arange(2 * dofs))
                and all(np.isfinite(value).all() for value in (k, powers, residuals))
                and np.all((residuals >= 0) & (residuals <= 1e-8)), "Invalid complete SAFE mode partition or residuals.")
        groups, selected = {}, []
        for name, sign, partition, expected in (("incoming", 1, positive, positive_roots),
                                                 ("reflection", -1, negative, negative_roots)):
            roots = partition[propagating[partition]]
            require(np.array_equal(roots, expected) and len(roots) == 5 and len(np.unique(roots)) == 5,
                    f"The complete five-channel {name} group differs from its receipt.")
            require(np.all(k[roots].imag == 0) and np.all(k[roots].real != 0)
                    and np.allclose(powers[roots], sign, rtol=0, atol=1e-7), "Signed unit transport is invalid.")
            groups[name] = {"roots": roots.copy(), "k": k[roots].real.copy()}
            selected.extend(roots.tolist())
        displacement, force = arrays["displacements"], arrays["axial_forces"]
        require(displacement.shape == force.shape == (dofs, 2 * dofs)
                and displacement.dtype == force.dtype == np.complex128, "SAFE field dimensions or precision differ.")
        for value in groups.values():
            roots = value["roots"]
            u, f = displacement[:, roots], force[:, roots]
            actual = math.pi * frequency_hz * np.sum((u.conj() * f).imag, axis=0)
            require(np.isfinite(u).all() and np.isfinite(f).all()
                    and np.allclose(actual, powers[roots], rtol=0, atol=1e-7), "Saved axial-force transport differs.")
            value["displacements"] = u.copy()
            value["powers_w"] = actual.tolist()
    forward, backward = groups["incoming"], groups["reflection"]
    partner = []
    for wave in forward["k"]:
        choices = np.flatnonzero(np.isclose(backward["k"], -wave, rtol=1e-12, atol=0))
        require(len(choices) == 1, "A signed channel lacks a unique reverse-wavenumber partner.")
        partner.append(int(backward["roots"][choices[0]]))
    require(len(set(partner)) == 5, "Reverse-wavenumber association is not one-to-one.")
    groups["reverse_partners"] = dict(zip(map(int, forward["roots"]), partner, strict=True))
    return groups


def match_group(first, second, mass):
    """Retain the entire overlap diagnostic even when existing individual-match gates fail."""
    a, b = first["displacements"], second["displacements"]
    require(a.ndim == b.ndim == 2 and a.shape[0] == b.shape[0] == mass.shape[0]
            and mass.shape[0] == mass.shape[1] and np.isfinite(mass).all(), "Invalid matching fields or section mass.")
    cross = a.conj().T @ mass @ b
    norm_a = np.sqrt(np.einsum("ni,ni->i", a.conj(), mass @ a).real)
    norm_b = np.sqrt(np.einsum("ni,ni->i", b.conj(), mass @ b).real)
    require(np.all(norm_a > 0) and np.all(norm_b > 0), "A saved mode has nonpositive mass norm.")
    cross /= np.outer(norm_a, norm_b)
    row, column = linear_sum_assignment(-abs(cross))
    diagnostics = {"status": "unresolved", "reference_roots": np.asarray(first["roots"]).tolist(),
        "candidate_roots_unordered": np.asarray(second["roots"]).tolist(),
        "full_mass_overlap_matrix": abs(cross).tolist(),
        "full_normalized_complex_overlap": np.stack((cross.real, cross.imag), axis=-1).tolist(),
        "proposed_candidate_order": column.tolist(),
        "proposed_reference_order": row.tolist(),
        "proposed_mass_overlaps": abs(cross[row, column]).tolist()}
    try:
        require(len(first["roots"]) == len(second["roots"]) == 5, "Changed five-channel count breaks correspondence.")
        for value in (first, second):
            wave = np.asarray(value["k"])
            require(np.all(np.diff(np.sort(wave)) > matching.K_GAP_TOL * np.max(abs(wave))),
                    "Degenerate wavenumbers break individual correspondence.")
        objects = [matching.Modes(np.asarray(value["roots"]), np.asarray(value["k"]),
                                  value["displacements"][None, :, :]) for value in (first, second)]
        order, phase, accepted = matching.align(*objects, mass, None, False)
        diagnostics.update(status="matched", candidate_order=order.tolist(),
                           candidate_gauge_complex=np.stack((phase.real, phase.imag), axis=-1).tolist(), **accepted)
    except ValueError as error:
        diagnostics["reason"] = str(error)
    return diagnostics


def relation_from_modes(reference, candidate, mass):
    groups = {name: match_group(reference[name], candidate[name], mass) for name in ("incoming", "reflection")}
    groups["transmission"] = groups["incoming"].copy()
    status = "matched" if all(value["status"] == "matched" for value in groups.values()) else "unresolved"
    if status == "matched":
        forward = dict(zip(groups["incoming"]["reference_roots"], groups["incoming"]["candidate_roots"], strict=True))
        backward = dict(zip(groups["reflection"]["reference_roots"], groups["reflection"]["candidate_roots"], strict=True))
        consistent = all(backward[reverse] == candidate["reverse_partners"][forward[root]]
                         for root, reverse in reference["reverse_partners"].items())
        if not consistent:
            status = "unresolved"
        pairing = {"status": "consistent" if consistent else "unresolved",
                   "reference_forward_to_reverse": reference["reverse_partners"],
                   "candidate_forward_to_reverse": candidate["reverse_partners"]}
    else:
        pairing = {"status": "unresolved", "reason": "Individual signed correspondence failed."}
    return {"status": status, "matched_channels": groups, "reverse_partner_consistency": pairing}


def build_relation(workspace=ROOT):
    inputs = {}
    paths = {name: pinned(workspace, name, inputs) for name in
             ("safe_receipt", "section", "interval_inputs", "candidate_point", "candidate_modes", "interval_verification")}
    receipt, point, accepted, ledger = (json_read(paths[name]) for name in
                                      ("safe_receipt", "candidate_point", "interval_inputs", "interval_verification"))
    validate_metadata(receipt, point, accepted, ledger)
    roots = lambda sign: [row["root_id"] for row in point["metrics"]["open_channels"] if row["direction"] == sign]
    reference = read_signed_modes(paths["section"], REFERENCE_HZ, receipt["incident_port_roots"], receipt["backward_port_roots"])
    candidate = read_signed_modes(paths["candidate_modes"], CANDIDATE_HZ, roots("positive"), roots("negative"))
    with np.load(paths["section"], allow_pickle=False) as arrays:
        mass = arrays["M"]
    require(mass.shape == (PORT_DOFS, PORT_DOFS) and np.allclose(mass, mass.T, rtol=0, atol=1e-25),
            "The pinned consistent section mass differs.")
    result = relation_from_modes(reference, candidate, mass)
    return {**AUTHOR, "schema_version": 1, "phase": "saved_section_relation", **result,
        "reference_frequency_hz": REFERENCE_HZ, "candidate_frequency_hz": CANDIDATE_HZ,
        "trace_dofs": PORT_DOFS, "inputs": inputs,
        "section_array_sha256": accepted["identity"]["archive"]["section_array_sha256"],
        "trace_sha256": receipt["trace_sha256"],
        "reference_signed_power_w": {name: reference[name]["powers_w"] for name in ("incoming", "reflection")},
        "candidate_signed_power_w": {name: candidate[name]["powers_w"] for name in ("incoming", "reflection")},
        "matching_method": {"minimum_mass_overlap": matching.MIN_OVERLAP, "relative_wavenumber_gap": matching.K_GAP_TOL,
            "one_to_one_assignment": "Maximum sum of normalized mass overlaps within each signed group.",
            "gauge": "candidate aligned coefficient = outgoing overlap phase * raw coefficient / incoming overlap phase",
            "matching_source": link(Path(matching.__file__), workspace)},
        "body_solve_performed": False, "safe_eigensolve_performed": False,
        "continuous_branch_identity_established": False, "scope": SCOPE}


def load_relation(path, workspace=ROOT):
    """Reuse the accepted endpoint relation without repeating saved-section analysis."""
    require(sha(path) == ACCEPTED_RELATION_SHA256, "The accepted executed relation artifact SHA-256 differs.")
    relation = json_read(path)
    require(relation.get("content_sha256") == digest({key: value for key, value in relation.items()
                                                       if key != "content_sha256"}), "The saved relation content digest differs.")
    require(relation["phase"] == "saved_section_relation" and relation["status"] == "matched"
            and relation["reference_frequency_hz"] == REFERENCE_HZ
            and relation["candidate_frequency_hz"] == CANDIDATE_HZ
            and relation["trace_dofs"] == PORT_DOFS
            and relation["reverse_partner_consistency"]["status"] == "consistent"
            and relation["matching_method"]["minimum_mass_overlap"] == matching.MIN_OVERLAP
            and relation["matching_method"]["relative_wavenumber_gap"] == matching.K_GAP_TOL,
            "An unresolved or different saved relation cannot be used for a device comparison.")
    require(relation["producer"]["sha256"] == RELATION_PRODUCER_SHA256,
            "The retained executed relation producer differs.")
    for role, expected in relation["inputs"].items():
        require(role in PINS and expected["sha256"] == PINS[role][1], "An accepted relation input role differs.")
        actual = (Path(workspace) / expected["path"]).resolve()
        require(actual == (Path(workspace) / PINS[role][0]).resolve() and sha(actual) == expected["sha256"]
                and actual.stat().st_size == expected["bytes"], "A saved relation input changed.")
    required = {"safe_receipt", "section", "interval_inputs", "candidate_point", "candidate_modes", "interval_verification"}
    require(set(relation["inputs"]) == required, "The saved relation lacks required provenance roles.")
    for group in relation["matched_channels"].values():
        require(group["status"] == "matched" and len(group["reference_roots"]) == len(group["candidate_roots"]) == 5
                and len(set(group["reference_roots"])) == len(set(group["candidate_roots"])) == 5
                and min(group["mass_overlaps"]) >= matching.MIN_OVERLAP, "The saved matched group differs.")
    return relation


def parse_response(summary, totals, coefficients, roots, reverse, frequency_hz):
    """Join every raw coefficient by its signed identity, independent of CSV row order."""
    require(summary["frequency_hz"] == frequency_hz and summary["incident_side"] == "left"
            and summary["incident_mode_indices"] == roots and summary["num_incident_channels"] == 5
            and summary["power_balance_passed"] is True and len(totals) == 5 and len(coefficients) == 50,
            "Response metadata, count, frequency or signed input order differs.")
    total_by_order = {}
    for row in totals:
        order = int(row["incident_mode_order"])
        values = [float(row[name]) for name in ("reflection", "transmission")]
        require(order in range(1, 6) and order not in total_by_order
                and math.isclose(float(row["frequency_GHz"]) * 1e9, frequency_hz, rel_tol=1e-14)
                and np.isfinite(values).all() and min(values) >= 0 and abs(sum(values) - 1) < .01,
                "Total response row duplicates a channel or has invalid frequency/powers.")
        total_by_order[order] = values
    entries = {}
    for row in coefficients:
        kind, incoming, outgoing = row["kind"], int(row["incident_port_index"]), int(row["outgoing_port_index"])
        order = int(row["incident_mode_order"])
        require(kind in ("reflection", "transmission") and order in range(1, 6) and roots[order - 1] == incoming
                and outgoing in (reverse if kind == "reflection" else roots), "A raw coefficient uses the wrong signed role.")
        real, imag, power = (float(row[name]) for name in ("amplitude_real", "amplitude_imag", "power_fraction"))
        require(np.isfinite([real, imag, power]).all() and power >= 0
                and math.isclose(power, real**2 + imag**2, rel_tol=0, abs_tol=TOLERANCE),
                "A raw coefficient power differs from its complex amplitude.")
        key = kind, incoming, outgoing
        require(key not in entries, "A signed raw coefficient is duplicated.")
        entries[key] = {"kind": kind, "incident_root": incoming, "outgoing_root": outgoing,
                        "amplitude": [real, imag], "power": power}
    rows = []
    for i, (incoming, same_reverse) in enumerate(zip(roots, reverse, strict=True), 1):
        row = {"root": incoming, "reverse_root": same_reverse}
        for kind, outgoing_roots, same, quantity in (("reflection", reverse, same_reverse, "R"),
                                                     ("transmission", roots, incoming, "T")):
            group = [entries[kind, incoming, outgoing] for outgoing in outgoing_roots]
            power = math.fsum(value["power"] for value in group)
            require(abs(power - total_by_order[i][0 if quantity == "R" else 1]) <= TOLERANCE,
                    "Raw coefficient powers differ from the response total.")
            row[quantity] = power
            row["converted_" + quantity] = math.fsum(value["power"] for value in group if value["outgoing_root"] != same)
        row["total_converted_power"] = row["converted_R"] + row["converted_T"]
        rows.append(row)
    return {"run_hash": summary["run_hash"], "frequency_hz": frequency_hz, "rows": rows,
            "coefficients": list(entries.values())}


def power_change(reference, candidate):
    delta = candidate - reference
    denominator = max(abs(reference), RELATIVE_FLOOR)
    return {"reference_power": reference, "candidate_power": candidate, "power_change": delta,
            "absolute_power_change": abs(delta), "relative_power_change": delta / denominator,
            "relative_denominator": denominator, "denominator_floor_applied": abs(reference) < RELATIVE_FLOOR}


def compare_responses(reference, candidate, relation, *, selected_pair=PAIR):
    require(relation["status"] == "matched", "Unresolved endpoint correspondence breaks the comparison.")
    matches = relation["matched_channels"]
    root_maps = {name: dict(zip(value["reference_roots"], value["candidate_roots"], strict=True))
                 for name, value in matches.items()}
    incoming = root_maps["incoming"]
    lookups = [{(row["kind"], row["incident_root"], row["outgoing_root"]): row
                for row in response["coefficients"]} for response in (reference, candidate)]
    gauges = {name: dict(zip(value["reference_roots"], value["candidate_gauge_phase_rad"], strict=True))
              for name, value in matches.items()}
    coefficients = []
    for kind in ("reflection", "transmission"):
        for reference_in, candidate_in in incoming.items():
            for reference_out, candidate_out in root_maps[kind].items():
                a = lookups[0][kind, reference_in, reference_out]
                b = lookups[1][kind, candidate_in, candidate_out]
                raw_a, raw_b = complex(*a["amplitude"]), complex(*b["amplitude"])
                aligned = raw_b * np.exp(1j * (gauges[kind][reference_out] - gauges["incoming"][reference_in]))
                delta = aligned - raw_a
                coefficients.append({"kind": kind, "reference_incident_root": reference_in,
                    "candidate_incident_root": candidate_in, "reference_outgoing_root": reference_out,
                    "candidate_outgoing_root": candidate_out, "reference_raw_amplitude": a["amplitude"],
                    "candidate_raw_amplitude": b["amplitude"], "candidate_gauge_aligned_amplitude": [aligned.real, aligned.imag],
                    "aligned_complex_delta": [delta.real, delta.imag], "absolute_complex_change": abs(delta),
                    **power_change(a["power"], b["power"])})
    reference_rows = {row["root"]: row for row in reference["rows"]}
    candidate_rows = {row["root"]: row for row in candidate["rows"]}
    rows = []
    for root, candidate_root in incoming.items():
        a, b = reference_rows[root], candidate_rows[candidate_root]
        require(root_maps["reflection"][a["reverse_root"]] == b["reverse_root"], "Signed reverse correspondence differs.")
        for quantity in ("R", "T", "converted_R", "converted_T", "total_converted_power"):
            rows.append({"quantity": quantity, "reference_incident_root": root, "candidate_incident_root": candidate_root,
                         "reference_outgoing_root": "", "candidate_outgoing_root": "", **power_change(a[quantity], b[quantity])})
    pair = []
    for incident, outgoing in selected_pair:
        selected = [row for row in coefficients if row["kind"] == "reflection"
                    and row["reference_incident_root"] == incident and row["reference_outgoing_root"] == outgoing]
        require(len(selected) == 1, "The preselected signed conversion pair is absent.")
        pair.append(selected[0])
        rows.append({key: value for key, value in {"quantity": "selected_pair_reflection", **selected[0]}.items()
                     if key in rows[0]})
    pair_sum = power_change(math.fsum(row["reference_power"] for row in pair),
                            math.fsum(row["candidate_power"] for row in pair))
    rows.append({"quantity": "selected_pair_two_drive_power_sum", "reference_incident_root": "",
                 "candidate_incident_root": "", "reference_outgoing_root": "", "candidate_outgoing_root": "", **pair_sum})
    return {"per_drive": rows[:25], "selected_pair": pair, "selected_pair_two_drive_power_sum": pair_sum,
            "coefficients": coefficients, "raw_responses": {"reference": reference, "candidate": candidate}}, rows


def validate_candidate_verification(verification, summary_path, workspace=ROOT):
    require(verification["status"] == "point_verified_within_saved_discretization"
            and verification["frequency_hz"] == CANDIDATE_HZ
            and verification["public_summary_sha256"] == sha(summary_path)
            and all(verification[key] is True for key in ("original_equations_passed", "physical_nodal_force_power_passed",
                    "all_modal_coefficients_passed", "full_p2_field_identity_passed", "coherent_left_incidence_passed")),
            "The candidate lacks a bound successful independent equation/flux/field/coherent replay.")
    for role, filename, expected in (("verifier", "replay_lawrence_frequency_point_AP_pnc.py", REPLAY_SOURCE_SHA256),
                                     ("shared_replay", "replay_lawrence_device_point_AP_pnc.py", SHARED_REPLAY_SHA256)):
        recorded = verification["implementation"][role]
        source = Path(workspace) / "tools" / filename
        require(recorded["path"].replace("\\", "/").rsplit("/", 1)[-1] == filename
                and recorded["sha256"] == expected == sha(source)
                and recorded["bytes"] == source.stat().st_size,
                f"The independent replay {role} is not the reviewed source.")
    published = verification["public_summary"]
    require(published["path"].replace("\\", "/").rsplit("/", 1)[-1] == "summary_AP_pnc.json"
            and published["sha256"] == sha(summary_path) and published["bytes"] == Path(summary_path).stat().st_size,
            "The independent replay public-summary artifact differs.")


def validate_common_device(reference, candidate):
    for name in ("geometry", "mesh", "material"):
        require(reference["configuration"][name] == candidate["configuration"][name], f"The physical device {name} changed.")
    for key in ("mesh_hash", "mesh_sha256"):
        require(reference["mesh_checkpoint"][key] == candidate["mesh_checkpoint"][key], "The saved body discretization changed.")
    for name in ("ndof", "hole_count", "port_dofs", "incident_side"):
        require(reference[name] == candidate[name], f"The device {name} differs.")
    for stem in (*matching.CORE, "layered", "assembly_batched"):
        name = f"src/phononic/{stem}_AP_pnc.py"
        require(reference["implementation"]["source_sha256"][name] == candidate["implementation"]["source_sha256"][name],
                f"The physical numerical source changed: {name}.")
    require(candidate["ndof"] == 616239 and candidate["hole_count"] == 26 and candidate["port_dofs"] == PORT_DOFS
            and candidate["status"] == "completed_unverified" and candidate["power_balance_passed"] is True
            and candidate["configuration"]["analysis"] == {"kind": "scattering", "boundary": "matched_ports",
                   "frequency_GHz": CANDIDATE_HZ / 1e9, "incident_side": "left"}
            and candidate["checkpointed_scattering"]["reference_planes_m"] == [0., 1.665e-5]
            and candidate["checkpointed_scattering"]["phase_adjustment_applied"] is False,
            "The bounded candidate device, frequency or original end-plane convention differs.")


def build_comparison(relation_path, candidate_directory, verification_path, workspace=ROOT, *, verification_sha256):
    require(matching.is_sha256(verification_sha256) and sha(verification_path) == verification_sha256,
            "The accepted independent point-replay receipt SHA-256 differs.")
    relation = load_relation(relation_path, workspace)
    inputs = {"mode_relation": link(relation_path, workspace), "candidate_verification": link(verification_path, workspace)}
    documents = {}
    for role in ("reference_summary", "reference_totals", "reference_coefficients", "conversion_verification"):
        documents[role] = pinned(workspace, role, inputs)
    reference, candidate_path = json_read(documents["reference_summary"]), Path(candidate_directory) / "summary_AP_pnc.json"
    candidate = json_read(candidate_path)
    validate_candidate_verification(json_read(verification_path), candidate_path, workspace)
    validate_common_device(reference, candidate)
    inputs["candidate_summary"] = link(candidate_path, workspace)
    candidate_files = {}
    for role, name in (("totals", "scattering_AP_pnc.csv"), ("coefficients", "scattering_channels_AP_pnc.csv")):
        path = Path(candidate_directory) / name
        require(sha(path) == candidate["output_sha256"][name], "A candidate response CSV checksum differs.")
        inputs["candidate_" + role] = link(path, workspace)
        candidate_files[role] = path
    point = json_read(Path(workspace) / PINS["candidate_point"][0])
    candidate_roots = {sign: [row["root_id"] for row in point["metrics"]["open_channels"] if row["direction"] == sign]
                       for sign in ("positive", "negative")}
    reverse_maps = relation["reverse_partner_consistency"]
    modes = ((reference["incident_mode_indices"], [reverse_maps["reference_forward_to_reverse"][str(root)]
                                                  for root in reference["incident_mode_indices"]]),
             (candidate_roots["positive"], [reverse_maps["candidate_forward_to_reverse"][str(root)]
                                            for root in candidate_roots["positive"]]))
    def read_csv(path):
        with Path(path).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    responses = []
    for summary, roots, reverse, frequency, total_path, coefficient_path in (
            (reference, *modes[0], REFERENCE_HZ, documents["reference_totals"], documents["reference_coefficients"]),
            (candidate, *modes[1], CANDIDATE_HZ, candidate_files["totals"], candidate_files["coefficients"])):
        responses.append(parse_response(summary, read_csv(total_path), read_csv(coefficient_path), roots, reverse, frequency))
    compared, rows = compare_responses(*responses, relation)
    previous = json_read(documents["conversion_verification"])["preserved_scientific_status"]
    require(previous["mesh_pair_passed"] is False and previous["complex_passed"] is False,
            "The preserved failed 1 GHz thickness status changed.")
    return {**AUTHOR, "schema_version": 1, "phase": "device_frequency_comparison", "status": "compared_finite_frequency_variation",
        "reference_frequency_hz": REFERENCE_HZ, "candidate_frequency_hz": CANDIDATE_HZ,
        "inputs": inputs, "matched_channels": relation["matched_channels"], **compared,
        "executed_relation_producer": relation["producer"],
        "comparison_producer_policy": "The pinned relation retains its executed source identity; this comparison records its own current producer separately.",
        "power_units": "Dimensionless power fraction per 1 W incident drive; the pair sum combines two separate drives.",
        "relative_change_definition": "(candidate - reference) / max(abs(reference), relative_denominator_floor)",
        "relative_denominator_floor": RELATIVE_FLOOR, "physical_significance_threshold": None,
        "preserved_one_ghz_thickness_status": previous, "other_prior_failed_spatial_comparisons_remain_unresolved": True,
        "spatial_convergence_established": False, "frequency_convergence_established": False,
        "continuous_branch_identity_established": False, "phase_adjustment_applied": False, "scope": SCOPE}, rows


def write_outputs(output, result, rows=()):
    output = Path(output)
    relation = result["phase"] == "saved_section_relation"
    json_name = "mode_relation_AP_pnc.json" if relation else "device_frequency_comparison_AP_pnc.json"
    csv_name = "device_frequency_comparison_AP_pnc.csv"
    payload = {**result, "producer": link(Path(__file__))}
    payload["content_sha256"] = digest(payload)
    files = {json_name: json.dumps(payload, indent=2, allow_nan=False) + "\n"}
    downloads = [report_link("Complete source-bound JSON", json_name, download=True)]
    if rows:
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        files[csv_name] = stream.getvalue()
        downloads.append(report_link("Endpoint response comparison (CSV)", csv_name, download=True))
    body = f'<p><strong>Status:</strong> {escape(result["status"].replace("_", " "))}</p>'
    body += f'<p class="sources">{" · ".join(downloads)}</p>'
    if relation:
        title = "Signed port-mode relation between 887.5 MHz and 1 GHz"
        scope = ("Sampled endpoint correspondence on the saved h80/z6 end sections at the original x = 0 and 16.65 µm planes. "
                 "The relation uses signed port power and mass overlap. Continuous branch identity remains unresolved.")
        for name, group in result["matched_channels"].items():
            body += f"<h2>{escape(name.title())}: {escape(group['status'])}</h2>"
            body += f"<p>{escape(group.get('reason', 'Sampled endpoint correspondence accepted.'))}</p>"
            body += "<p>Rows: " + escape(str(group["reference_roots"])) + "; columns: " + escape(str(group["candidate_roots_unordered"])) + "</p>"
            columns = "".join(f'<th scope="col">{root}</th>' for root in group["candidate_roots_unordered"])
            cells = "".join(f'<tr><th scope="row">{root}</th>' + "".join(f"<td>{v:.10g}</td>" for v in row) + "</tr>"
                            for root, row in zip(group["reference_roots"], group["full_mass_overlap_matrix"], strict=True))
            body += scroll_table(f'<table><thead><tr><th scope="col">1 GHz root</th>{columns}</tr></thead>'
                                 f'<tbody>{cells}</tbody></table>', f"{name.title()} mass overlap matrix")
    else:
        title = "Saved device responses at 887.5 MHz and 1 GHz"
        scope = (SCOPE + " Signed section-mode matching pairs the endpoint roots. Raw amplitudes retain their native "
                 "modal gauges. The JSON and CSV distinguish native values and gauge-aligned differences.")
    if rows:
        columns = "".join(f'<th scope="col">{escape(key)}</th>' for key in rows[0])
        cells = "".join("<tr>" + "".join(f"<td>{escape(str(row[key]))}</td>" for key in rows[0]) + "</tr>" for row in rows)
        body += scroll_table(f"<table><thead><tr>{columns}</tr></thead><tbody>{cells}</tbody></table>",
                             "Endpoint response comparisons")
    navigation = available_link(output, ROOT / "figures/README.md", "Device figures and evidence")
    compact = available_link(output, ROOT / "verification/device_frequency_8875_AP_pnc.json", "Compact archived frequency evidence (JSON)")
    files["report_AP_pnc.html"] = report_page(title, "device_frequency", body + f'<p>{compact}</p>',
                                                 navigation=navigation, scope=scope)
    output.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        path = output / name
        require(not path.exists() or path.read_bytes() == text.encode("utf-8"),
                "Existing scientific comparison differs; choose a new output directory.")
    for name, text in files.items():
        path = output / name
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(text)
    return output / json_name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    relation = subparsers.add_parser("relation", help="Read only the pinned saved sections before any body job.")
    relation.add_argument("--workspace", type=Path, default=ROOT)
    relation.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("compare", help="Compare the independently accepted new response with the pinned 1 GHz response.")
    final.add_argument("--workspace", type=Path, default=ROOT)
    final.add_argument("--relation", type=Path, required=True)
    final.add_argument("--candidate", type=Path, required=True)
    final.add_argument("--verification", type=Path, required=True)
    final.add_argument("--verification-sha256", required=True,
                       help="SHA-256 of the independently accepted final point-replay receipt.")
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "relation":
        result, rows = build_relation(args.workspace), ()
    else:
        result, rows = build_comparison(args.relation, args.candidate, args.verification, args.workspace,
                                       verification_sha256=args.verification_sha256)
    path = write_outputs(args.output, result, rows)
    print(f"{result['status']}: {path.resolve()}")
    return 2 if result["status"] == "unresolved" else 0


if __name__ == "__main__":
    raise SystemExit(main())
