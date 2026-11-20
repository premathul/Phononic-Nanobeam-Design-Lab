"""Compare one archived 887.5 MHz z4/z6 control. Athul Prem | _AP_pnc.

The relation phase reads only saved section arrays. The comparison phase reads
small, receipt-bound response tables; neither phase performs a numerical solve.
"""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from phononic.report_style_AP_pnc import available_link, report_link, report_page, scroll_table
from phononic.workflow_AP_pnc import digest
from tools import compare_scattering_AP_pnc as matching
from tools import compare_device_frequency_AP_pnc as frequency
from tools import diagnose_port_polarization_AP_pnc as section_reader

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
FREQUENCY, ANCHOR_FREQUENCY = 887_500_000, 1_000_000_000
PAIR_RELATIVE_LIMIT, SUPPRESSION_RATIO_LIMIT = .10, .01
PLANES = [0., 1.665e-5]
TRACE_SHA = "2752f71b459282bb33610b39ce99fd25eee6784a510e4c6013415a9357e6fd7b"
BODY_STAGE_SHA = "1c50971e27bfc52e63000c9d25df100e21bbe28b7fa47d7549f82cf465531a78"
BASE = "results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc/"
STUDY = "results/runs/device_frequency_8875_AP_pnc/"
CONTROL = "results/runs/device_frequency_thickness_8875_AP_pnc/"
# These two immutable pre-body artifacts precede a receipt-consumer correction.
# This exception does not admit other receipts or unretained source revisions.
EXECUTED_RELATION_SHA = "5ee5c8006803ee34ae7ef554588fdfa1f887c065c032c924cb79a6cc0558b9b4"
RELATION_PRODUCER_SHA = "1fb709dfacb1c9e2a45d2ac4d43fce7ab5a5c471021f6026bb85406936ecb08b"
RELATION_PRODUCER_BYTES = 40466
RELATION_PRODUCER_SNAPSHOT = CONTROL + "local_checks/compare_frequency_thickness_relation_AP_pnc.py"
PREBODY_RECEIPT_SHA = "f45245144be6035c72ffee5ced2d96aa349b939bde89f9bb36f6ca82550bdca4"
PREBODY_PRODUCER_SHA = "34e2307663a9ea2c3cb864562a58559a6b94c8710219ea17fd305013e0110572"
PREBODY_PRODUCER_BYTES = 40611
PREBODY_PRODUCER_SNAPSHOT = CONTROL + "local_checks/replay_lawrence_frequency_z4_prebody_AP_pnc.py"
PINS = {
    "z4_section": ("results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr/arrays_AP_pnc.npz", "2a2aa9347921ac6940f851f2eb53bc06480a28a6cf67c27a000faa357b1404e8"),
    "z4_section_receipt": ("results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr/checkpoint_AP_pnc.json", "677c64a816ea42a8b37e1358d5549d63ecaeec7623a9366139ba64d301bbea4e"),
    "z6_section": (BASE + "thickness-h80-safe-1335308/ports_AP_pnc.npz", "a13a58e47dee91e4c7d75bb9ec77feb7656ae674392c2f1d7485129da76d0e6a"),
    "z6_section_receipt": (BASE + "thickness-h80-safe-1335308/thickness_h80_safe_AP_pnc.json", "b5c0d73656e90aa5ba312939c0ef25633af5a775974f5fc778031584d55c78f3"),
    "one_ghz_thickness_comparison": (BASE + "comparisons/scattering-b06d34228c560e43/summary_AP_pnc.json", "8a5aea4238d3f09bdd7e97f82cd73d2b74f6a64846ca946633aec08d0b6dede3"),
    "one_ghz_conversion_verification": ("verification/one_ghz_conversion_AP_pnc.json", "f9a72bdcb173c31b96fc0d6d21e589ba9a3a94aa3c9d57877bfd7d6fd90459fd"),
    "accepted_z6_relation": (STUDY + "relation/mode_relation_AP_pnc.json", "6792e815654f5cc882310ab51a70f80ff5a0280e7f5db62df2ec1a30feaf8e65"),
    "accepted_z6_frequency_comparison": (STUDY + "comparison/device_frequency_comparison_AP_pnc.json", "b69aa76de99a01f60fad8f1640170d4fa266ccec312061050894aa823eaa4813"),
    "accepted_z6_modes": ("results/runs/port_count_interval_AP_pnc/study/modes-887500000_AP_pnc.npz", "93e4e089e6b329244b204ca10e3612653824b1e569476a51ea1924654a83b81c"),
    "accepted_z6_point_replay": (STUDY + "independent_point_AP_pnc.json", "7ca962aa197087aa167e1dfb3a74fdaee1b822b2d3c7d2a6e4cf5f3c936a27d7"),
}
# Live reader hashes include the 2026-11-20 presentation update. Numerical
# functions are unchanged; archived producer/receipt identities remain separate.
HELPERS = {
    "matching_source": (matching, "c677eae9a1078b90638950e01107de64bf96a44a27b24b675d09b74353c2a2f8"),
    "section_reader_source": (section_reader, "367eb1fa5ed0296df548716cb8eb02e422dd639a23f8ad434dc0e357af5d1e79"),
    "response_reader_source": (frequency, "04e7f7531e551c4172d0dd69c89775f5557e4f392bf047260d1b0b3ed75cec58"),
}
SCOPE = ("One finite joint body/port thickness-discretization increment at 887.5 MHz. "
         "The 10% directed-power increment and 99% suppression-retention targets are observational criteria, "
         "not accuracy estimates, validated floors, uncertainty bounds or convergence claims. Original end planes "
         "and all earlier failed spatial checks are retained. Endpoint matches do not prove continuous branches.")
require, sha = frequency.require, frequency.sha


def artifact(path, workspace=ROOT):
    return frequency.link(Path(path), workspace)


def read_json(path):
    require(Path(path).stat().st_size <= 4 * 2**20, "A bounded metadata input exceeds 4 MiB.")
    return json.loads(Path(path).read_bytes())


def basename(link):
    return str(link.get("path", link.get("file", ""))).replace("\\", "/").rsplit("/", 1)[-1]


def same_artifact(first, second):
    return (basename(first) == basename(second)
            and all(first.get(key) == second.get(key) for key in ("sha256", "bytes")))


def checked_file(path, expected_sha):
    path = Path(path).resolve(strict=True)
    require(matching.is_sha256(expected_sha) and path.is_file() and sha(path) == expected_sha,
            "An externally accepted artifact SHA-256 differs.")
    return path


def load_pins(workspace=ROOT):
    root, paths, inputs = Path(workspace).resolve(), {}, {}
    for role, (relative, expected) in PINS.items():
        path = checked_file(root / relative, expected)
        require(path.is_relative_to(root), "A pinned scientific role escapes the workspace.")
        paths[role], inputs[role] = path, artifact(path, root)
    for role, (module, expected) in HELPERS.items():
        path = checked_file(module.__file__, expected)
        inputs[role] = artifact(path, root)
    return paths, inputs


def checked_safe(path, expected_sha, runner_sha, workspace=ROOT):
    path = checked_file(path, expected_sha)
    value = read_json(path)
    runner = checked_file(Path(workspace) / "tools/run_lawrence_frequency_z4_AP_pnc.py", runner_sha)
    require(value["status"] == "safe_ready" and value["frequency_hz"] == FREQUENCY
            and value["displacement_dofs"] == 426627 and value["discrete_roots"] == 1350
            and value["positive_open_channels"] == value["negative_open_channels"] == 5
            and value["cutoff_status"] == "solver_resolved_at_sample"
            and value["degeneracy_status"] == "none_detected"
            and value["coincident_propagating_root_groups"] == []
            and value["reference_planes_m"] == PLANES and value["trace_sha256"] == TRACE_SHA,
            "The new five-channel z4 SAFE receipt differs from the bounded control.")
    require(value["driver"]["sha256"] == runner_sha and value["body_stage"]["sha256"] == BODY_STAGE_SHA
            and value["safe_section"]["sha256"] == PINS["z4_section"][1]
            and value["section_receipt"]["sha256"] == PINS["z4_section_receipt"][1],
            "The reviewed runner or archived body/section provenance differs.")
    require(all(value[key] is False for key in ("body_assembly_performed", "section_assembly_performed",
            "mesh_generated", "anchor_eigensolve_performed", "factorization_performed")),
            "The section sample changed archived inputs or performed body computation.")
    mode_path = checked_file(path.parent / basename(value["safe_modes"]), value["safe_modes"]["sha256"])
    require(same_artifact(artifact(mode_path), value["safe_modes"]), "The new SAFE mode artifact differs.")
    return value, mode_path, artifact(runner, workspace)


def read_modes(path, frequency_hz, dofs, positive_roots, negative_roots, *, legacy=False):
    """Read the complete partition and ten signed fields, without body arrays."""
    require(Path(path).stat().st_size <= 64 * 2**20, "Section arrays exceed 64 MiB.")
    with np.load(path, allow_pickle=False) as data:
        k = data["k_rad_m" if legacy else "wavenumbers_rad_m"]
        power = data["signed_power_w" if legacy else "powers_w"]
        opened, residual = data["propagating"], data["relative_residuals"]
        positive, negative = data["positive_indices"], data["negative_indices"]
        require(legacy or float(data["frequency_hz"]) == frequency_hz, "SAFE frequency differs.")
        require(k.shape == power.shape == opened.shape == residual.shape == (2 * dofs,)
                and opened.dtype == np.bool_ and positive.shape == negative.shape == (dofs,)
                and positive.dtype.kind in "iu" and negative.dtype.kind in "iu"
                and np.array_equal(np.sort(np.r_[positive, negative]), np.arange(2 * dofs))
                and np.isfinite(k).all() and np.isfinite(residual).all()
                and np.all((residual >= 0) & (residual <= 1e-8)), "Invalid complete SAFE partition or residuals.")
        u, force = data["displacements"], data["axial_forces"]
        require(u.shape == force.shape == (dofs, 2 * dofs) and u.dtype == force.dtype == np.complex128,
                "The SAFE field precision or dimensions differ.")
        groups = {}
        for group, sign, ids, expected in (("incoming", 1, positive, positive_roots),
                                           ("reflection", -1, negative, negative_roots)):
            roots = ids[opened[ids]]
            require(roots.tolist() == list(expected) and len(roots) == len(set(roots)) == 5,
                    "A complete signed five-channel group differs from its receipt.")
            transport = math.pi * frequency_hz * np.sum((u[:, roots].conj() * force[:, roots]).imag, axis=0)
            require(np.all(k[roots].imag == 0) and np.all(k[roots].real != 0)
                    and np.isfinite(u[:, roots]).all() and np.isfinite(force[:, roots]).all()
                    and np.allclose(power[roots], sign, rtol=0, atol=1e-7)
                    and np.allclose(transport, sign, rtol=0, atol=1e-7), "Invalid signed unit-power fields.")
            groups[group] = {"roots": roots.copy(), "k": k[roots].real.copy(),
                             "displacements": u[:, roots].copy(), "powers_w": transport.tolist()}
    partners = {}
    for root, wave in zip(groups["incoming"]["roots"], groups["incoming"]["k"], strict=True):
        selected = groups["reflection"]["roots"][np.isclose(groups["reflection"]["k"], -wave, rtol=1e-12, atol=0)]
        require(len(selected) == 1, "An incoming channel lacks a unique reverse-wavenumber partner.")
        partners[int(root)] = int(selected[0])
    require(len(set(partners.values())) == 5, "Reverse partners are not bijective.")
    groups["reverse_partners"] = partners
    return groups


def match_edge(first_section, first, second_section, second):
    """Use the established richer P2 trace and preserve unresolved diagnostics."""
    project_first = first_section.basis.N <= second_section.basis.N
    source, target = (first_section, second_section) if project_first else (second_section, first_section)
    same = (np.array_equal(source.basis.doflocs, target.basis.doflocs)
            and np.array_equal(source.basis.mesh.t, target.basis.mesh.t))
    probe = None if same else matching.interpolation(source, target)
    groups = {}
    for name in ("incoming", "reflection"):
        a, b = first[name], second[name]
        aa, bb = a["displacements"][first_section.trace_to_scalar], b["displacements"][second_section.trace_to_scalar]
        if probe is not None:
            if project_first:
                aa = np.stack([probe @ component for component in aa])
            else:
                bb = np.stack([probe @ component for component in bb])
        cross = np.einsum("cni,cnj->ij", aa.conj(), target.mass @ bb)
        norms = [np.sqrt(np.einsum("cni,cni->i", value.conj(), target.mass @ value).real) for value in (aa, bb)]
        require(all(np.all(value > 0) for value in norms), "A signed field has nonpositive trace mass norm.")
        cross /= np.outer(*norms)
        rows, order = linear_sum_assignment(-abs(cross))
        result = {"status": "unresolved", "reference_roots": a["roots"].tolist(),
                  "candidate_roots_unordered": b["roots"].tolist(),
                  "full_mass_overlap_matrix": abs(cross).tolist(),
                  "full_normalized_complex_overlap": np.stack((cross.real, cross.imag), axis=-1).tolist(),
                  "proposed_candidate_order": order.tolist(), "proposed_mass_overlaps": abs(cross[rows, order]).tolist()}
        try:
            for value in (a, b):
                require(np.all(np.diff(np.sort(value["k"])) > matching.K_GAP_TOL * max(abs(value["k"]))),
                        "Degenerate signed roots break individual matching.")
            objects = (matching.Modes(a["roots"], a["k"], aa), matching.Modes(b["roots"], b["k"], bb))
            order, phase, accepted = matching.align(*objects, target.mass, None, False)
            result.update(status="matched", candidate_order=order.tolist(),
                          candidate_gauge_complex=np.stack((phase.real, phase.imag), axis=-1).tolist(), **accepted)
        except ValueError as error:
            result["reason"] = str(error)
        groups[name] = result
    groups["transmission"] = dict(groups["incoming"])
    status = "matched" if all(row["status"] == "matched" for row in groups.values()) else "unresolved"
    pairing = {"status": "unresolved", "reference_forward_to_reverse": first["reverse_partners"],
               "candidate_forward_to_reverse": second["reverse_partners"]}
    if status == "matched":
        maps = {name: root_map(groups[name]) for name in ("incoming", "reflection")}
        consistent = all(maps["reflection"][reverse] == second["reverse_partners"][maps["incoming"][root]]
                         for root, reverse in first["reverse_partners"].items())
        pairing["status"] = "consistent" if consistent else "unresolved"
        if not consistent:
            status = "unresolved"
    return {"status": status, "matched_channels": groups, "reverse_partner_consistency": pairing,
            "projection": "Original fields interpolated onto the richer P2 trace; no propagation phase removed."}


def root_map(group):
    return dict(zip(group["reference_roots"], group["candidate_roots"], strict=True))


def permutation_square(old_cross, old_frequency, new_frequency, new_cross):
    """Check only channel identities around the four endpoints, not gauge holonomy."""
    rows, consistent = {}, True
    try:
        for group in ("incoming", "reflection"):
            a, b, c, d = [root_map(edge["matched_channels"][group])
                          for edge in (old_cross, old_frequency, new_frequency, new_cross)]
            require(set(a) == set(c) and set(a.values()) == set(b)
                    and set(c.values()) == set(d), "The endpoint square has incompatible signed root sets.")
            rows[group] = [{"z4_one_ghz_root": root, "z6_one_ghz_root": a[root],
                            "z4_8875_root": c[root], "z6_8875_via_z6": b[a[root]],
                            "z6_8875_via_z4": d[c[root]]} for root in a]
            consistent &= all(row["z6_8875_via_z6"] == row["z6_8875_via_z4"] for row in rows[group])
    except (ValueError, KeyError) as error:
        return {"status": "unresolved", "reason": str(error), "signed_paths": rows, "gauge_closure_required": False}
    return {"status": "consistent" if consistent else "unresolved", "signed_paths": rows,
            "gauge_closure_required": False}


def build_relation(safe_path, safe_sha, runner_sha, workspace=ROOT):
    paths, inputs = load_pins(workspace)
    safe, mode_path, runner = checked_safe(safe_path, safe_sha, runner_sha, workspace)
    inputs.update(candidate_safe_receipt=artifact(safe_path, workspace), candidate_safe_modes=artifact(mode_path, workspace))
    old_cross = read_json(paths["one_ghz_thickness_comparison"])
    old_frequency = read_json(paths["accepted_z6_relation"])
    require(old_cross["status"] == "compared" and old_cross["complex_passed"] is False
            and old_cross["mesh_pair_passed"] is False and old_frequency["status"] == "matched"
            and old_frequency["reverse_partner_consistency"]["status"] == "consistent",
            "An accepted anchor relation or retained failed spatial status differs.")
    _, z4_section = section_reader.read_section(paths["z4_section"], "z4")
    _, z6_section = section_reader.read_section(paths["z6_section"], "z6")
    old4 = read_modes(paths["z4_section"], ANCHOR_FREQUENCY, 675,
                      old_cross["matched_channels"]["incoming"]["reference_roots"],
                      old_cross["matched_channels"]["reflection"]["reference_roots"], legacy=True)
    new4 = read_modes(mode_path, FREQUENCY, 675, safe["incident_port_roots"], safe["backward_port_roots"])
    new6 = read_modes(paths["accepted_z6_modes"], FREQUENCY, 975,
                      old_frequency["matched_channels"]["incoming"]["candidate_roots_unordered"],
                      old_frequency["matched_channels"]["reflection"]["candidate_roots_unordered"])
    edges = {"cross_z_8875": match_edge(z4_section, new4, z6_section, new6),
             "z4_frequency": match_edge(z4_section, old4, z4_section, new4)}
    square = permutation_square(old_cross, old_frequency, edges["z4_frequency"], edges["cross_z_8875"])
    accepted = all(edge["status"] == "matched" for edge in edges.values()) and square["status"] == "consistent"
    return {**AUTHOR, "schema_version": 1, "phase": "saved_section_relation",
            "status": "signed_section_relation_accepted" if accepted else "signed_section_relation_unresolved",
            "frequency_hz": FREQUENCY, "reference_frequency_hz": ANCHOR_FREQUENCY,
            "candidate_safe_receipt": inputs["candidate_safe_receipt"], "candidate_safe_modes": inputs["candidate_safe_modes"],
            "positive_open_channels": 5, "negative_open_channels": 5, "overlap_gate": matching.MIN_OVERLAP,
            "relative_wavenumber_gap": matching.K_GAP_TOL, "reviewed_runner": runner,
            "inputs": inputs, "edges": edges, "root_permutation_square": square,
            "spatial_convergence_established": False, "frequency_convergence_established": False,
            "continuous_branch_identity_established": False, "body_solve_performed": False,
            "body_assembly_performed": False, "safe_eigensolve_performed": False,
            "scope": SCOPE}


def accepted_relation(path, expected_sha, workspace=ROOT):
    path = checked_file(path, expected_sha)
    value = read_json(path)
    compatibility = None
    if value["producer"]["sha256"] != sha(__file__):
        require(expected_sha == EXECUTED_RELATION_SHA
                and value["producer"]["sha256"] == RELATION_PRODUCER_SHA
                and value["producer"]["bytes"] == RELATION_PRODUCER_BYTES,
                "The production relation was not made by the current or exact retained executed producer.")
        snapshot = checked_file(Path(workspace) / RELATION_PRODUCER_SNAPSHOT, RELATION_PRODUCER_SHA)
        require(snapshot.stat().st_size == RELATION_PRODUCER_BYTES, "The retained relation producer size differs.")
        compatibility = {"accepted_relation": artifact(path, workspace), "executed_producer": value["producer"],
                         "retained_executed_producer": artifact(snapshot, workspace),
                         "current_consumer": artifact(__file__, workspace),
                         "reason": "Receipt-consumer correction only; the exact accepted relation and executed producer are unchanged."}
    require(value["content_sha256"] == digest({key: item for key, item in value.items() if key != "content_sha256"})
            and value["phase"] == "saved_section_relation" and value["status"] == "signed_section_relation_accepted"
            and value["frequency_hz"] == FREQUENCY and value["reference_frequency_hz"] == ANCHOR_FREQUENCY
            and value["positive_open_channels"] == value["negative_open_channels"] == 5
            and value["overlap_gate"] == matching.MIN_OVERLAP
            and value["root_permutation_square"]["status"] == "consistent",
            "The independently accepted production relation differs or is unresolved.")
    for role, (_, expected) in PINS.items():
        require(value["inputs"][role]["sha256"] == expected, "A production relation anchor role differs.")
    for edge in value["edges"].values():
        require(edge["status"] == "matched" and edge["reverse_partner_consistency"]["status"] == "consistent",
                "A production relation edge is unresolved.")
    return {**value, "accepted_producer_compatibility": compatibility}


def positive_ratio(numerator, denominator, label):
    require(math.isfinite(numerator) and numerator >= 0 and math.isfinite(denominator) and denominator > 0,
            f"{label} requires an exact positive accepted denominator; no floor is substituted.")
    return numerator / denominator


def pair_targets(z4_power, z6_power, z4_anchor_power, z6_anchor_power):
    """Both directions must pass individually; averaging cannot hide a failure."""
    require(all(len(values) == 2 for values in (z4_power, z6_power, z4_anchor_power, z6_anchor_power)),
            "Exactly two preselected directed conversions are required.")
    increments, suppression = [], []
    for direction in range(2):
        a, b = z4_power[direction], z6_power[direction]
        relative = positive_ratio(abs(a - b), b, "Directed finite increment")
        increments.append({"direction": direction + 1, "z4_power": a, "accepted_z6_power": b,
                           "signed_z4_minus_z6_power": a - b, "absolute_power_change": abs(a - b),
                           "relative_absolute_power_change": relative, "exact_denominator": b,
                           "limit": PAIR_RELATIVE_LIMIT, "passed": relative <= PAIR_RELATIVE_LIMIT})
        for name, power, anchor in (("z4", a, z4_anchor_power[direction]), ("z6", b, z6_anchor_power[direction])):
            ratio = positive_ratio(power, anchor, "Suppression retention")
            suppression.append({"direction": direction + 1, "resolution": name, "power_8875": power,
                                "power_one_ghz": anchor, "ratio": ratio, "decrease_fraction": 1 - ratio,
                                "limit": SUPPRESSION_RATIO_LIMIT, "passed": ratio <= SUPPRESSION_RATIO_LIMIT})
    return {"directed_increment_rows": increments, "directed_increment_passed": all(row["passed"] for row in increments),
            "suppression_retention_rows": suppression, "suppression_retention_passed": all(row["passed"] for row in suppression),
            "denominator_floor_applied": False, "interpretation": "Finite observational targets only; no accuracy or uncertainty bound."}


def complex_pair_change(reference, raw_candidate, aligned_candidate):
    """Zero amplitudes retain absolute changes and explicitly undefined relative/phase values."""
    def describe(value):
        delta = value - reference
        return {"candidate_amplitude": [value.real, value.imag], "complex_delta": [delta.real, delta.imag],
                "absolute_complex_change": abs(delta),
                "relative_denominator_amplitude": abs(reference),
                "relative_absolute_complex_change": abs(delta) / abs(reference) if reference != 0 else None,
                "phase_change_rad": float(np.angle(value / reference)) if reference != 0 and value != 0 else None}
    return {"reference_amplitude": [reference.real, reference.imag], "raw": describe(raw_candidate),
            "gauge_aligned": describe(aligned_candidate), "additional_phase_gate_applied": False}


def checked_replay(path, expected_sha, source_sha, workspace=ROOT, purpose="point"):
    path = checked_file(path, expected_sha)
    value = read_json(path)
    verifier = checked_file(Path(workspace) / "tools/replay_lawrence_frequency_z4_AP_pnc.py", source_sha)
    compatibility = None
    producer = value["implementation"]["verifier"]
    if producer["sha256"] != sha(verifier):
        require(purpose == "prebody" and expected_sha == PREBODY_RECEIPT_SHA
                and producer["sha256"] == PREBODY_PRODUCER_SHA and producer["bytes"] == PREBODY_PRODUCER_BYTES,
                "The independent replay was not produced by the current or exact retained prebody source.")
        snapshot = checked_file(Path(workspace) / PREBODY_PRODUCER_SNAPSHOT, PREBODY_PRODUCER_SHA)
        require(snapshot.stat().st_size == PREBODY_PRODUCER_BYTES, "The retained prebody producer size differs.")
        compatibility = {"accepted_prebody_receipt": artifact(path, workspace), "executed_producer": producer,
                         "retained_executed_producer": artifact(snapshot, workspace),
                         "current_consumer": artifact(verifier, workspace),
                         "reason": "The point consumer corrects export receipt handling; accepted SAFE and relation analysis are reused unchanged."}
    return {**value, "accepted_producer_compatibility": compatibility}


def response_inputs(summary_path, workspace, inputs, prefix):
    summary_path = Path(summary_path).resolve(strict=True)
    summary = read_json(summary_path)
    inputs[prefix + "_summary"] = artifact(summary_path, workspace)
    tables = []
    for role, name in (("totals", "scattering_AP_pnc.csv"), ("coefficients", "scattering_channels_AP_pnc.csv")):
        path = checked_file(summary_path.parent / name, summary["output_sha256"][name])
        inputs[prefix + "_" + role] = artifact(path, workspace)
        with path.open(newline="", encoding="utf-8") as stream:
            tables.append(list(csv.DictReader(stream)))
    return summary, tables


def parsed_response(summary, tables, edge, label, frequency_hz):
    roots = summary["incident_mode_indices"]
    require(set(roots) == set(edge["matched_channels"]["incoming"][label + "_roots"]),
            "The raw response and signed relation use different incoming channels.")
    partners = edge["reverse_partner_consistency"][label + "_forward_to_reverse"]
    reverse = [partners.get(str(root), partners.get(root)) for root in roots]
    return frequency.parse_response(summary, *tables, roots, reverse, frequency_hz)


def build_comparison(relation_path, relation_sha, prebody_path, prebody_sha, candidate_directory,
                     point_path, point_sha, replay_source_sha, workspace=ROOT):
    paths, inputs = load_pins(workspace)
    relation = accepted_relation(relation_path, relation_sha, workspace)
    prebody = checked_replay(prebody_path, prebody_sha, replay_source_sha, workspace, "prebody")
    point = checked_replay(point_path, point_sha, replay_source_sha, workspace)
    require(prebody["status"] == "independently_verified_safe"
            and prebody["relation_status"] == "matched"
            and prebody["relation_replay"]["status"] == "independently_verified_signed_relation"
            and prebody["assigned_root_square_consistent"] is True
            and same_artifact(prebody["relation"], artifact(relation_path, workspace))
            and same_artifact(prebody["safe_receipt"], relation["candidate_safe_receipt"])
            and same_artifact(prebody["safe_modes"], relation["candidate_safe_modes"]),
            "The combined independent section/relation receipt does not bind this accepted relation.")
    require(prebody["implementation"]["comparison"]["sha256"] == relation["producer"]["sha256"]
            and prebody["implementation"]["runner"]["sha256"] == relation["reviewed_runner"]["sha256"]
            and same_artifact(point["prebody_replay"], artifact(prebody_path, workspace)),
            "The accepted source identities or independent point lineage differ.")
    historical = prebody.get("accepted_producer_compatibility")
    if historical is not None:
        compatibility = point["prebody_producer_compatibility"]
        require(compatibility["status"] == "verified_exact_saved_prebody_producer"
                and same_artifact(compatibility["accepted_prebody_receipt"], historical["accepted_prebody_receipt"])
                and same_artifact(compatibility["accepted_producer"], historical["executed_producer"])
                and same_artifact(compatibility["retained_producer"], historical["retained_executed_producer"])
                and same_artifact(compatibility["consumer"], historical["current_consumer"]),
                "The independent point replay does not retain the exact accepted prebody producer compatibility.")
    inputs.update(production_relation=artifact(relation_path, workspace), prebody_replay=artifact(prebody_path, workspace),
                  point_replay=artifact(point_path, workspace))
    accepted_frequency = read_json(paths["accepted_z6_frequency_comparison"])
    candidate_summary_path = Path(candidate_directory) / "summary_AP_pnc.json"
    z4_summary, z4_tables = response_inputs(candidate_summary_path, workspace, inputs, "z4_8875")
    require(point["status"] == "point_verified_within_saved_discretization" and point["frequency_hz"] == FREQUENCY
            and point["public_summary_sha256"] == sha(candidate_summary_path)
            and same_artifact(point["public_summary"], artifact(candidate_summary_path, workspace))
            and point["relative_target"] == 1e-10
            and point["relative_residual_limit"] == point["backward_residual_limit"] == 1e-9
            and 0 <= point["maximum_relative_residual"] <= 1e-9
            and 0 <= point["maximum_backward_residual"] <= 1e-9
            and all(point[key] is True for key in ("original_equations_passed", "physical_nodal_force_power_passed",
                "all_modal_coefficients_passed", "full_p2_field_identity_passed", "coherent_left_incidence_passed")),
            "The candidate does not have the accepted original-equation/flux/field/coherent replay.")
    accepted_path = Path(workspace) / accepted_frequency["inputs"]["candidate_summary"]["path"]
    require(sha(accepted_path) == accepted_frequency["inputs"]["candidate_summary"]["sha256"],
            "The accepted z6 public-summary identity differs.")
    z6_summary, z6_tables = response_inputs(accepted_path, workspace, inputs, "z6_8875")
    for key in ("geometry", "material"):
        require(z4_summary["configuration"][key] == z6_summary["configuration"][key], "The common physical device changed.")
    mesh4, mesh6 = (summary["configuration"]["mesh"] for summary in (z4_summary, z6_summary))
    require({k: v for k, v in mesh4.items() if k != "thickness_layers"}
            == {k: v for k, v in mesh6.items() if k != "thickness_layers"}
            and mesh4["thickness_layers"] == 4 and mesh6["thickness_layers"] == 6,
            "The fixed-planar joint thickness/port control changed other mesh parameters.")
    require(z4_summary["ndof"] == 426627 and z6_summary["ndof"] == 616239
            and z4_summary["port_dofs"] == 675 and z6_summary["port_dofs"] == 975,
            "The archived body/section dimensions differ.")
    for summary in (z4_summary, z6_summary):
        require(summary["status"] == "completed_unverified" and summary["hole_count"] == 26
                and summary["checkpointed_scattering"]["reference_planes_m"] == PLANES
                and summary["checkpointed_scattering"]["phase_adjustment_applied"] is False,
                "A completed device or its original reference planes differs.")
    require(z4_summary["implementation"]["packages"] == z6_summary["implementation"]["packages"],
            "Numerical package versions changed between the two responses.")
    for stem in (*matching.CORE, "layered", "assembly_batched"):
        key = f"src/phononic/{stem}_AP_pnc.py"
        require(z4_summary["implementation"]["source_sha256"][key]
                == z6_summary["implementation"]["source_sha256"][key], "A physical numerical source changed.")
    edge = relation["edges"]["cross_z_8875"]
    responses = [parsed_response(summary, tables, edge, label, FREQUENCY)
                 for summary, tables, label in ((z4_summary, z4_tables, "reference"), (z6_summary, z6_tables, "candidate"))]
    z6_pairs = [(row["candidate_incident_root"], row["candidate_outgoing_root"])
                for row in accepted_frequency["selected_pair"]]
    inverse_in = {v: k for k, v in root_map(edge["matched_channels"]["incoming"]).items()}
    inverse_out = {v: k for k, v in root_map(edge["matched_channels"]["reflection"]).items()}
    selected = [(inverse_in[incident], inverse_out[outgoing]) for incident, outgoing in z6_pairs]
    compared, _ = frequency.compare_responses(*responses, edge, selected_pair=selected)
    old = read_json(paths["one_ghz_conversion_verification"])
    old_pairs = old["selected_conversion_pairs"]
    anchors = [[row["power_fraction"] for row in old_pairs[name]["directions"]] for name in ("h80_z4", "h80_z6")]
    require([(row["incident_root"], row["outgoing_root"]) for row in old_pairs["h80_z6"]["directions"]]
            == [(row["reference_incident_root"], row["reference_outgoing_root"]) for row in accepted_frequency["selected_pair"]],
            "The preselected 1 GHz directed pair changed.")
    z4_frequency = relation["edges"]["z4_frequency"]["matched_channels"]
    for anchor, pair in zip(old_pairs["h80_z4"]["directions"], selected, strict=True):
        require((root_map(z4_frequency["incoming"])[anchor["incident_root"]],
                 root_map(z4_frequency["reflection"])[anchor["outgoing_root"]]) == pair,
                "The direct z4 endpoint relation selects another pair.")
    pair_rows = compared["selected_pair"]
    exact_accepted_z6 = [row["candidate_power"] for row in accepted_frequency["selected_pair"]]
    require([row["candidate_power"] for row in pair_rows] == exact_accepted_z6
            and anchors[1] == [row["reference_power"] for row in accepted_frequency["selected_pair"]],
            "An exact accepted z6 directed-power denominator differs.")
    # The reusable parser reports a guarded relative change. This milestone uses
    # exact denominators and leaves zero-reference relative changes undefined.
    for row in (*compared["per_drive"], *compared["coefficients"], compared["selected_pair_two_drive_power_sum"]):
        denominator = row["reference_power"]
        row.update(relative_power_change=row["power_change"] / denominator if denominator > 0 else None,
                   relative_denominator=denominator, denominator_floor_applied=False)
    for row in pair_rows:
        row["complex_changes"] = complex_pair_change(complex(*row["reference_raw_amplitude"]),
            complex(*row["candidate_raw_amplitude"]), complex(*row["candidate_gauge_aligned_amplitude"]))
    targets = pair_targets([row["reference_power"] for row in pair_rows],
                           exact_accepted_z6, *anchors)
    power = float(max(row["absolute_power_change"] for row in compared["per_drive"] if row["quantity"] in ("R", "T")))
    complex_change = float(max(row["absolute_complex_change"] for row in compared["coefficients"]))
    input_passed = all(summary["power_balance_passed"] is True for summary in (z4_summary, z6_summary))
    transmission_phases = []
    for row in compared["coefficients"]:
        if row["kind"] == "transmission" and row["reference_incident_root"] == row["reference_outgoing_root"]:
            first, second = complex(*row["reference_raw_amplitude"]), complex(*row["candidate_gauge_aligned_amplitude"])
            transmission_phases.append({"z4_incident_root": row["reference_incident_root"],
                "z6_incident_root": row["candidate_incident_root"],
                "phase_change_rad": float(np.angle(second / first)) if first != 0 and second != 0 else None})
    return {**AUTHOR, "schema_version": 1, "phase": "device_thickness_comparison", "status": "compared_finite_thickness_increment",
            "frequency_hz": FREQUENCY, "reference_resolution": "z4", "candidate_resolution": "z6",
            "inputs": inputs, "matched_channels": edge["matched_channels"], **compared,
            "executed_relation_source": relation["producer"],
            "accepted_relation_producer_compatibility": relation.get("accepted_producer_compatibility"),
            "accepted_prebody_producer_compatibility": prebody.get("accepted_producer_compatibility"),
            "input_power_checks_passed": input_passed, "power_change_limit": matching.LIMIT,
            "complex_change_limit": matching.LIMIT, "maximum_power_change": power, "maximum_complex_change": complex_change,
            "power_passed": power < matching.LIMIT, "complex_passed": complex_change < matching.LIMIT,
            "mesh_pair_passed": input_passed and power < matching.LIMIT and complex_change < matching.LIMIT,
            "same_channel_transmission_phase_changes": transmission_phases,
            "pair_specific_targets": targets, "pair_complex_phase_gate_applied": False,
            "power_units": "Dimensionless power fraction for each separate 1 W incident drive.",
            "signed_power_increment": "z6 minus z4 in coefficient rows; pair target rows explicitly use z4 minus accepted z6.",
            "preserved_one_ghz_thickness_status": old["preserved_scientific_status"],
            "spatial_convergence_established": False, "frequency_convergence_established": False,
            "continuous_branch_identity_established": False, "phase_adjustment_applied": False,
            "other_prior_failed_spatial_comparisons_remain_unresolved": True, "scope": SCOPE}


def write_outputs(output, result, workspace=ROOT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = "mode_relation_AP_pnc.json" if result["phase"] == "saved_section_relation" else "frequency_thickness_comparison_AP_pnc.json"
    payload = {**result, "producer": artifact(__file__, workspace)}
    payload["content_sha256"] = digest(payload)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path = output / name
    if path.exists():
        require(path.read_text(encoding="utf-8") == text, "Existing scientific output differs; choose a new output directory.")
    else:
        path.write_text(text, encoding="utf-8")
    downloads = [available_link(output, path, "Complete source-bound JSON", download=True)]
    if result["phase"] == "device_thickness_comparison":
        rows = result["coefficients"]
        flat_rows = []
        for row in rows:
            flat = {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
            for key, value in row.items():
                if isinstance(value, list) and len(value) == 2:
                    flat.update({key + "_real": value[0], key + "_imag": value[1]})
            flat_rows.append(flat)
        columns = list(flat_rows[0])
        csv_path = output / "frequency_thickness_comparison_AP_pnc.csv"
        from io import StringIO
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
        csv_text = stream.getvalue()
        if csv_path.exists():
            require(csv_path.read_text(encoding="utf-8") == csv_text.replace("\r\n", "\n"), "Existing coefficient CSV differs.")
        else:
            csv_path.write_text(csv_text, encoding="utf-8", newline="")
        downloads.append(available_link(output, csv_path, "Complete coefficient table (CSV)", download=True))
    body = f'<p><strong>Status:</strong> {escape(result["status"].replace("_", " "))}</p>'
    body += f'<p class="sources">{" · ".join(downloads)}</p>'
    if result["phase"] == "saved_section_relation":
        title = "Signed port-mode relation for the 887.5 MHz thickness-layer control"
        scope = ("Sampled signed correspondence on the h80 four- and six-layer sections at fixed physical thickness. "
                 "Continuous branch identity remains unresolved.")
        body += "<p>Root-permutation square: " + escape(result["root_permutation_square"]["status"]) + ". The relation uses root identity without imposing gauge closure.</p>"
    else:
        title = "887.5 MHz thickness-layer and port sensitivity at fixed thickness"
        scope = "The h80 four-to-six-layer change refines the body and compatible ports at fixed 200 nm physical thickness. " + SCOPE
        body += "<p>Whole-response power / complex increments: " + f"{result['maximum_power_change']:.12g} / {result['maximum_complex_change']:.12g}. "
        power_status = "passed" if result["power_passed"] else "failed"
        complex_status = "passed" if result["complex_passed"] else "failed"
        body += f"Power check {power_status}. Complex check {complex_status}.</p>"
        targets = result["pair_specific_targets"]
        directed = "".join(f"<tr><td>{row['direction']}</td><td>{row['z4_power']:.12g}</td><td>{row['accepted_z6_power']:.12g}</td>"
                           f"<td>{row['relative_absolute_power_change']:.12g}</td><td>{row['passed']}</td></tr>"
                           for row in targets["directed_increment_rows"])
        body += "<h2>Directed-power increments</h2>" + scroll_table(
            "<table><thead><tr><th>Direction</th><th>z4 power</th><th>z6 power</th><th>|ΔP|/Pz6</th><th>≤10%</th></tr></thead>"
            f"<tbody>{directed}</tbody></table>", "Directed-power increments")
        ratios = "".join(f"<tr><td>{row['resolution']}</td><td>{row['direction']}</td><td>{row['ratio']:.12g}</td><td>{row['passed']}</td></tr>"
                         for row in targets["suppression_retention_rows"])
        body += "<h2>Suppression-retention ratios</h2>" + scroll_table(
            "<table><thead><tr><th>Resolution</th><th>Direction</th><th>P8875/P1GHz</th><th>≤0.01</th></tr></thead>"
            f"<tbody>{ratios}</tbody></table>", "Suppression-retention ratios")
        body += "<p>The ratios use exact positive accepted denominators. The pair sum combines separate drives. Pair phase is diagnostic.</p>"
        coefficients = []
        number = lambda value: "undefined" if value is None else f"{value:.12g}"
        for row in result["selected_pair"]:
            change = row["complex_changes"]
            raw, aligned = change["raw"], change["gauge_aligned"]
            coefficients.append(f"<tr><td>{row['reference_incident_root']}→{row['reference_outgoing_root']}</td>"
                                f"<td>{row['candidate_incident_root']}→{row['candidate_outgoing_root']}</td>"
                                f"<td>{number(raw['absolute_complex_change'])}</td><td>{number(aligned['absolute_complex_change'])}</td>"
                                f"<td>{number(aligned['relative_absolute_complex_change'])}</td><td>{number(aligned['phase_change_rad'])}</td></tr>")
        body += "<h2>Selected complex coefficients</h2>" + scroll_table(
            "<table><thead><tr><th>z4 signed pair</th><th>z6 signed pair</th><th>Raw |ΔS|</th><th>Aligned |ΔS|</th>"
            "<th>Aligned |ΔS|/|Sz4|</th><th>Aligned Δphase (rad)</th></tr></thead>"
            f"<tbody>{''.join(coefficients)}</tbody></table>", "Selected complex coefficients")
        body += "<p>Relative complex changes use the displayed z4 coefficient magnitude. Raw phases retain their native modal gauges. Aligned changes use each direct signed relation.</p>"
    navigation = available_link(output, ROOT / "figures/README.md", "Device figures and evidence")
    report = report_page(title, "frequency_thickness_8875", body, navigation=navigation, scope=scope)
    report_path = output / "report_AP_pnc.html"
    if report_path.exists():
        require(report_path.read_text(encoding="utf-8") == report, "Existing report differs.")
    else:
        report_path.write_text(report, encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    relation = sub.add_parser("relation")
    relation.add_argument("--safe", type=Path, required=True)
    relation.add_argument("--safe-sha256", required=True)
    relation.add_argument("--runner-sha256", required=True)
    compare = sub.add_parser("compare")
    for name in ("relation", "prebody-replay", "point-replay"):
        compare.add_argument("--" + name, type=Path, required=True)
        compare.add_argument("--" + name + "-sha256", required=True)
    compare.add_argument("--replay-source-sha256", required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    for command in (relation, compare):
        command.add_argument("--workspace", type=Path, default=ROOT)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "relation":
        result = build_relation(args.safe, args.safe_sha256, args.runner_sha256, args.workspace)
    else:
        result = build_comparison(args.relation, args.relation_sha256, args.prebody_replay, args.prebody_replay_sha256,
            args.candidate, args.point_replay, args.point_replay_sha256, args.replay_source_sha256, args.workspace)
    print(write_outputs(args.output, result, args.workspace))
    return 0 if result["status"] != "signed_section_relation_unresolved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
