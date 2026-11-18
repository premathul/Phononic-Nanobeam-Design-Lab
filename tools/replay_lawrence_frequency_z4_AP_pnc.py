"""Independent 887.5 MHz z4 SAFE, signed-relation and body replay. Athul Prem | _AP_pnc.

The pre-body mode replays the entire new SAFE spectrum once, then verifies the
two new relation edges by explicit P2 integration and exhaustive assignment.
Body replay consumes that accepted receipt rather than repeating SAFE analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from itertools import permutations
import json
import os
from pathlib import Path
import sys
from time import monotonic

THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
if __name__ == "__main__":
    os.environ.update(THREADS)

import numpy as np
import psutil
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import replay_lawrence_device_point_AP_pnc as shared
from tools import replay_lawrence_z4_control_AP_pnc as old_z4
from tools import verify_port_polarization_AP_pnc as quadrature
from tools import verify_port_count_interval_AP_pnc as safe_replay

FREQUENCY, DOFS, PORT_DOFS, CHANNELS = 887_500_000, 426627, 675, 5
ARCHIVE, SECTION = old_z4.ARCHIVE, old_z4.SECTION
BODY_SHA, SECTION_SHA, SECTION_RECEIPT_SHA, TRACE_SHA = (
    old_z4.STAGE_SHA, old_z4.SECTION_SHA, old_z4.SECTION_RECEIPT_SHA, old_z4.TRACE_SHA)
SHARED_SHA = "83c98faae6f07071394fc472a1f44d96897258e09b6266db60402da63e06a6e0"
OLD_Z4_SHA = "cf2c4f441d23f9b772c4595c118f3bc3d88de14ea930422441f89c7515fc8ce7"
QUADRATURE_SHA = "d5cdd47c7df2de902224de6f7c70299862187fa372a0eecaaa38e358e4693779"
SAFE_REPLAY_SHA = "8488071bf9e19c53d29365f0806af0fefac2ba58528af7555a68b6420decc844"
FROZEN_RUNNER_SHA = "f849c02a536dd3d615d26a26ac1eb189bb0bd607c9f0ccda87cf79e8b86524de"
ACCEPTED_PREBODY_SHA = "f45245144be6035c72ffee5ced2d96aa349b939bde89f9bb36f6ca82550bdca4"
ACCEPTED_PREBODY_PRODUCER_SHA = "34e2307663a9ea2c3cb864562a58559a6b94c8710219ea17fd305013e0110572"
ACCEPTED_PREBODY_PRODUCER = ROOT / "results/runs/device_frequency_thickness_8875_AP_pnc/local_checks/replay_lawrence_frequency_z4_prebody_AP_pnc.py"
EXECUTION_KEYS = ("implementation", "driver", "hostname", "slurm_job_id", "numerical_thread_environment", "numerical_backend")
RUNNER = ROOT / "tools/run_lawrence_frequency_z4_AP_pnc.py"
COMPARATOR = ROOT / "tools/compare_frequency_thickness_AP_pnc.py"
Z6_SECTION = ROOT / "results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc/thickness-h80-safe-1335308/ports_AP_pnc.npz"
Z6_SECTION_SHA = "a13a58e47dee91e4c7d75bb9ec77feb7656ae674392c2f1d7485129da76d0e6a"
Z6_MODES = ROOT / "results/runs/port_count_interval_AP_pnc/study/modes-887500000_AP_pnc.npz"
Z6_MODES_SHA = "93e4e089e6b329244b204ca10e3612653824b1e569476a51ea1924654a83b81c"
OLD_THICKNESS = ROOT / "results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc/comparisons/scattering-b06d34228c560e43/summary_AP_pnc.json"
OLD_THICKNESS_SHA = "8a5aea4238d3f09bdd7e97f82cd73d2b74f6a64846ca946633aec08d0b6dede3"
Z6_RELATION = ROOT / "results/runs/device_frequency_8875_AP_pnc/relation/mode_relation_AP_pnc.json"
Z6_RELATION_SHA = "6792e815654f5cc882310ab51a70f80ff5a0280e7f5db62df2ec1a30feaf8e65"
RELATION_PINS = {
    "z4_section": (SECTION / "arrays_AP_pnc.npz", SECTION_SHA),
    "z4_section_receipt": (SECTION / "checkpoint_AP_pnc.json", SECTION_RECEIPT_SHA),
    "z6_section": (Z6_SECTION, Z6_SECTION_SHA),
    "z6_section_receipt": (Z6_SECTION.parent / "thickness_h80_safe_AP_pnc.json", "b5c0d73656e90aa5ba312939c0ef25633af5a775974f5fc778031584d55c78f3"),
    "one_ghz_thickness_comparison": (OLD_THICKNESS, OLD_THICKNESS_SHA),
    "one_ghz_conversion_verification": (ROOT / "verification/one_ghz_conversion_AP_pnc.json", "f9a72bdcb173c31b96fc0d6d21e589ba9a3a94aa3c9d57877bfd7d6fd90459fd"),
    "accepted_z6_relation": (Z6_RELATION, Z6_RELATION_SHA),
    "accepted_z6_frequency_comparison": (ROOT / "results/runs/device_frequency_8875_AP_pnc/comparison/device_frequency_comparison_AP_pnc.json", "b69aa76de99a01f60fad8f1640170d4fa266ccec312061050894aa823eaa4813"),
    "accepted_z6_modes": (Z6_MODES, Z6_MODES_SHA),
    "accepted_z6_point_replay": (ROOT / "results/runs/device_frequency_8875_AP_pnc/independent_point_AP_pnc.json", "7ca962aa197087aa167e1dfb3a74fdaee1b822b2d3c7d2a6e4cf5f3c936a27d7"),
    "matching_source": (ROOT / "tools/compare_scattering_AP_pnc.py", "148f7efa58ad668a7a57be9db68a3241ffb67ed94067499ce6dcd4a0f8c59313"),
    "section_reader_source": (ROOT / "tools/diagnose_port_polarization_AP_pnc.py", "367eb1fa5ed0296df548716cb8eb02e422dd639a23f8ad434dc0e357af5d1e79"),
    "response_reader_source": (ROOT / "tools/compare_device_frequency_AP_pnc.py", "36bb5effcd52ca7fde7da7f876acf8fb34bdaaf897bb4a6e8e7e28bb364ef623"),
}
SCOPE = dict.fromkeys(("body_solve_performed", "body_assembly_performed", "spatial_convergence_established",
                       "frequency_convergence_established", "continuous_branch_identity_established"), False)
FIELDS = old_z4.FIELDS
require = shared.require


def read(path):
    return json.loads(Path(path).read_bytes())


def artifact(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": shared.file_sha(path)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def same_artifact(first, second):
    def name(value):
        return value.get("path", value.get("file", "")).replace("\\", "/").rsplit("/", 1)[-1]
    return name(first) == name(second) and all(first[key] == second[key] for key in ("bytes", "sha256"))


def checked_link(path, link):
    require(same_artifact(artifact(path), link), f"Artifact name, size or SHA-256 differs: {path}")
    return Path(path)


def linked_basename(directory, link):
    return checked_link(Path(directory) / link.get("path", link.get("file", "")).replace("\\", "/").rsplit("/", 1)[-1], link)


def checked_helpers():
    for module, expected in ((shared, SHARED_SHA), (old_z4, OLD_Z4_SHA), (quadrature, QUADRATURE_SHA),
                             (safe_replay, SAFE_REPLAY_SHA)):
        require(shared.file_sha(Path(module.__file__)) == expected, "A frozen independent replay helper changed.")


def frequency_core(incident, returning):
    """Configure an isolated copy; preserve historical helpers and every gate."""
    checked_helpers()
    spec = importlib.util.spec_from_file_location("_frequency_z4_replay_gates_AP_pnc", Path(shared.__file__))
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.FREQUENCY, core.OMEGA = FREQUENCY, 2 * np.pi * FREQUENCY
    core.DOFS, core.PORT_DOFS, core.TRACE_SHA = DOFS, PORT_DOFS, TRACE_SHA
    core.INCIDENT, core.RETURNING = tuple(incident), tuple(returning)
    return core


def check_archive_identity(record, original, config, runner_sha):
    """Bind a new phase to the exact archived body and reviewed Linux source."""
    require(valid_sha(runner_sha) and record["driver"]["sha256"] == runner_sha
            and record["frequency_hz"] == FREQUENCY and record["displacement_dofs"] == DOFS
            and record["body_stage"]["sha256"] == BODY_SHA and record["trace_sha256"] == TRACE_SHA
            and record["reference_planes_m"] == list(shared.PLANES)
            and same_artifact(record["mesh"], original["mesh"])
            and same_artifact(record["mesh_summary"], original["mesh_checkpoint_summary"])
            and same_artifact(record["archived_safe_modes"], original["port_modes"])
            and all(same_artifact(record["array_files"][key], original[key]) for key in ("K", "M"))
            and record["safe_section"]["sha256"] == SECTION_SHA
            and record["section_receipt"]["sha256"] == SECTION_RECEIPT_SHA
            and all(record[key] is False for key in ("mesh_generated", "body_assembly_performed",
                     "section_assembly_performed", "factorization_performed")),
            "The z4 phase differs from its reviewed runner, exact archive or no-body scope.")
    expected = {**config, "analysis": {**config["analysis"], "frequency_GHz": FREQUENCY / 1e9},
                "resources": {"max_dofs": DOFS, "max_port_dofs": PORT_DOFS, "linear_solver": "direct"}}
    require(record["configuration"] == expected and record["body_assembly_source"] == original["source"],
            "The z4 phase changed archived physical inputs, configuration or assembly provenance.")
    old_z4.verify_numerical_sources(original["source"], record["implementation"])
    require(record["implementation"]["platform"].startswith("Linux-")
            and isinstance(record["slurm_job_id"], str) and record["slurm_job_id"].isascii()
            and record["slurm_job_id"].isdecimal() and record["numerical_thread_environment"] == THREADS,
            "The z4 phase lacks its single-threaded Linux Slurm provenance.")


def replay_modes(modes, section):
    """Reuse all accepted independent SAFE gates; enforce this sample's scope."""
    size = len(section["coordinates_yz_m"])
    k = modes["wavenumbers_rad_m"]
    plus, minus, opened = (modes[key] for key in ("positive_indices", "negative_indices", "propagating"))
    require(all(modes[key].dtype == np.complex128 for key in ("wavenumbers_rad_m", "displacements", "axial_forces")),
            "The complete saved SAFE fields must retain complex128 precision.")
    operators = {"K0": section["K0"], "M": section["mass_kg_per_m"], "Q": section["Q"], "G": section["G"]}
    metrics = safe_replay.replay_modes(operators, modes, FREQUENCY, length_m=1e-6,
                                       reference_speed_m_s=np.sqrt(128500000000.0 / 5323), batch_size=32)
    incident, returning = plus[opened[plus]], minus[opened[minus]]
    require(len(incident) == len(returning) == CHANNELS, "The z4 point does not have five channels per direction.")
    partners = reverse_partners(k, incident, returning)
    for roots in (incident, returning):
        wave = k[roots].real
        require(np.all(wave != 0) and np.all(np.diff(np.sort(wave)) > 1e-8 * np.max(abs(wave))),
                "The signed propagating channels are unresolved or degenerate.")
    return incident.tolist(), returning.tolist(), {**metrics, "roots_checked": 2 * size,
            "maximum_original_backward_residual": metrics["maximum_eigen_backward_residual"], "forward_to_reverse": partners}


def reverse_partners(waves, positive, negative):
    pairs = {}
    for root in positive:
        choices = [int(reverse) for reverse in negative if np.isclose(waves[root], -waves[reverse], rtol=1e-12, atol=0)]
        require(len(choices) == 1, "A signed root lacks a unique opposite-wavenumber partner.")
        pairs[int(root)] = choices[0]
    require(len(set(pairs.values())) == CHANNELS, "Opposite-wavenumber partners are not one-to-one.")
    return pairs


def load_archived_section():
    checked_link(SECTION / "arrays_AP_pnc.npz", {"path": "arrays_AP_pnc.npz", "bytes": 12973330, "sha256": SECTION_SHA})
    with np.load(SECTION / "arrays_AP_pnc.npz", allow_pickle=False) as arrays:
        return {key: arrays[key] for key in ("section_vertices_yz_m", "section_triangles", "coordinates_yz_m",
                    "component_dofs_xyz", "K0", "mass_kg_per_m", "Q", "G", "displacements", "k_rad_m",
                    "signed_power_w", "propagating", "positive_indices", "negative_indices", "relative_residuals")}


def load_modes(path):
    require(Path(path).stat().st_size <= 64 * 2**20, "SAFE archive exceeds the bounded 64 MiB input.")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in (*FIELDS, "frequency_hz")}


def replay_safe(safe_path, safe_sha, runner_sha):
    started = monotonic()
    checked_helpers()
    require(valid_sha(runner_sha) and shared.file_sha(RUNNER) == runner_sha, "The externally reviewed runner SHA-256 differs.")
    safe = shared.read_receipt(safe_path, safe_sha)
    original = shared.read_receipt(ARCHIVE / "stages_AP_pnc.json", BODY_SHA)
    config = read(checked_link(ARCHIVE / "configuration_AP_pnc.json", original["configuration"]))
    check_archive_identity(safe, original, config, runner_sha)
    section_receipt = shared.read_receipt(SECTION / "checkpoint_AP_pnc.json", SECTION_RECEIPT_SHA)
    require(safe["status"] == "safe_ready" and safe["safe_eigensolve_performed"] is True
            and safe["anchor_eigensolve_performed"] is False
            and section_receipt["arrays"]["sha256"] == SECTION_SHA
            and safe["discrete_roots"] == 2 * PORT_DOFS
            and safe["positive_open_channels"] == safe["negative_open_channels"] == CHANNELS
            and safe["cutoff_status"] == "solver_resolved_at_sample" and safe["degeneracy_status"] == "none_detected",
            "The SAFE receipt is incomplete, unresolved or outside the one-sample scope.")
    section = load_archived_section()
    modes_path = linked_basename(safe_path.parent, safe["safe_modes"])
    modes = load_modes(modes_path)
    require(modes["frequency_hz"].shape == () and modes["frequency_hz"].item() == FREQUENCY
            and len(section["coordinates_yz_m"]) == PORT_DOFS, "The raw z4 SAFE frequency or dimension differs.")
    incident, returning, metrics = replay_modes(modes, section)
    require(safe["incident_port_roots"] == incident and safe["backward_port_roots"] == returning,
            "The receipt roots differ from the complete independently replayed signed inventory.")
    result = {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "independently_verified_safe",
              "frequency_hz": FREQUENCY, "problem_dofs": DOFS, "port_dofs": PORT_DOFS,
              "safe_receipt": artifact(safe_path), "safe_modes": artifact(modes_path),
              "all_discrete_roots_replayed": 2 * PORT_DOFS, "positive_open_channels": CHANNELS,
              "negative_open_channels": CHANNELS, "incident_port_roots": incident, "backward_port_roots": returning,
              "port_replay": metrics, "reference_planes_m": list(shared.PLANES),
              "implementation": {"verifier": artifact(Path(__file__)), "runner": artifact(RUNNER),
                                  "shared_replay": artifact(Path(shared.__file__)),
                                  "quadrature": artifact(Path(quadrature.__file__)),
                                  "safe_replay": artifact(Path(safe_replay.__file__))},
              "elapsed_seconds": monotonic() - started, **SCOPE}
    return result, safe, section, modes


def signed_sections(section, modes, frequency, *, archived=False):
    waves = modes["k_rad_m" if archived else "wavenumbers_rad_m"]
    powers = modes["signed_power_w" if archived else "powers_w"]
    opened = modes["propagating"]
    positive, negative = (modes[key][opened[modes[key]]] for key in ("positive_indices", "negative_indices"))
    require(len(positive) == len(negative) == CHANNELS, "A relation endpoint does not have five signed channels.")
    if not archived:
        require(float(modes["frequency_hz"]) == frequency, "A relation endpoint uses another frequency.")
    results = {}
    topology = {"vertices_yz_m": section["section_vertices_yz_m"], "triangles": section["section_triangles"],
                "coordinates_yz_m": section["coordinates_yz_m"], "component_dofs_xyz": section["component_dofs_xyz"],
                "density_kg_m3": 5323.0}
    for label, ids, sign in (("incoming", positive, 1), ("reflection", negative, -1)):
        require(np.all(waves[ids].imag == 0) and np.all(waves[ids].real != 0)
                and np.allclose(powers[ids], sign, rtol=0, atol=1e-7), "A relation endpoint has invalid signed transport.")
        results[label] = {"roots": ids.tolist(), "k": waves[ids].real, "section": {
            **topology, "displacements": modes["displacements"][:, ids]}}
    results["reverse_partners"] = reverse_partners(waves, positive, negative)
    return results


def independent_edge_group(first, second):
    """Explicit P2 interpolation/integration on the richer trace; no saved mass."""
    source, target = first["section"], second["section"]
    require(len(source["coordinates_yz_m"]) <= len(target["coordinates_yz_m"]),
            "The direct edge must use its richer target trace.")
    projected = quadrature.interpolate_fields(source, target)
    integrated = quadrature.integrate_section(**dict(target, displacements=np.column_stack((projected, target["displacements"]))))
    gram = np.asarray(integrated["displacement_mass_gram_real"]) + 1j * np.asarray(integrated["displacement_mass_gram_imag"])
    cross = gram[:CHANNELS, CHANNELS:] / np.sqrt(np.outer(np.diag(gram)[:CHANNELS].real, np.diag(gram)[CHANNELS:].real))
    require(cross.shape == (CHANNELS, CHANNELS) and np.isfinite(cross).all(), "Invalid independently integrated overlap.")
    order = list(max(permutations(range(CHANNELS)), key=lambda choice: sum(abs(cross[i, j]) for i, j in enumerate(choice))))
    overlaps = np.asarray([abs(cross[i, j]) for i, j in enumerate(order)])
    valid = bool(np.all(overlaps >= .99))
    for i, j in enumerate(order):
        valid &= bool(overlaps[i] > np.max(np.delete(abs(cross[i]), j)) + 1e-8
                      and overlaps[i] > np.max(np.delete(abs(cross[:, j]), i)) + 1e-8)
    for fields in (first, second):
        wave = np.asarray(fields["k"])
        valid &= bool(np.all(np.diff(np.sort(wave)) > 1e-8 * np.max(abs(wave))))
    return {"status": "matched" if valid else "unresolved", "cross": cross, "order": order,
            "overlaps": overlaps, "gauges": np.asarray([cross[i, j] / overlaps[i] if overlaps[i] else 0 for i, j in enumerate(order)])}


def check_edge(published, reference, candidate):
    mappings, metrics = {}, {}
    for name in ("incoming", "reflection"):
        saved = published["matched_channels"][name]
        measured = independent_edge_group(reference[name], candidate[name])
        require(measured["status"] == "matched" and saved["status"] == "matched",
                "The independently integrated direct relation is unresolved.")
        order, cross, phase = measured["order"], measured["cross"], measured["gauges"]
        roots = [candidate[name]["roots"][i] for i in order]
        encoded = np.asarray(saved["full_normalized_complex_overlap"])
        require(encoded.shape == (CHANNELS, CHANNELS, 2)
                and np.max(abs(encoded[..., 0] + 1j * encoded[..., 1] - cross)) <= 1e-10
                and np.max(abs(np.asarray(saved["full_mass_overlap_matrix"]) - abs(cross))) <= 1e-10
                and saved["reference_roots"] == reference[name]["roots"]
                and saved["candidate_roots_unordered"] == candidate[name]["roots"]
                and saved["candidate_roots"] == roots and saved["candidate_order"] == order,
                "The direct relation overlaps or assigned roots differ from independent P2 replay.")
        gauge = np.asarray(saved["candidate_gauge_complex"])
        require(gauge.shape == (CHANNELS, 2)
                and np.max(abs(gauge[:, 0] + 1j * gauge[:, 1] - phase)) <= 1e-10
                and np.max(abs(np.exp(1j * np.asarray(saved["candidate_gauge_phase_rad"])) - phase)) <= 1e-10
                and np.max(abs(np.asarray(saved["mass_overlaps"]) - np.minimum(measured["overlaps"], 1))) <= 1e-10
                and np.array_equal(saved["reference_wavenumber_rad_m"], reference[name]["k"])
                and np.array_equal(saved["candidate_wavenumber_rad_m"], np.asarray(candidate[name]["k"])[order]),
                "The direct edge gauge, overlap or signed wavenumber differs.")
        mappings[name] = dict(zip(reference[name]["roots"], roots, strict=True))
        metrics[name] = {"minimum_mass_overlap": float(min(measured["overlaps"])),
                         "maximum_complex_overlap_difference": float(np.max(abs(encoded[..., 0] + 1j * encoded[..., 1] - cross)))}
    require(published["status"] == "matched" and published["matched_channels"]["transmission"] == published["matched_channels"]["incoming"]
            and published["reverse_partner_consistency"]["status"] == "consistent"
            and all(mappings["reflection"][reverse] == candidate["reverse_partners"][mappings["incoming"][root]]
                    for root, reverse in reference["reverse_partners"].items()),
            "The direct relation has inconsistent signed reverse partners or status.")
    for name, endpoint in (("reference", reference), ("candidate", candidate)):
        saved = published["reverse_partner_consistency"][f"{name}_forward_to_reverse"]
        require({int(key): int(value) for key, value in saved.items()} == endpoint["reverse_partners"],
                "The published reverse-partner map differs.")
    return mappings, metrics


def check_root_square(direct, cross, old_thickness, old_frequency, published=None):
    paths = {}
    for name in ("incoming", "reflection"):
        first, second = (document["matched_channels"][name] for document in (old_thickness, old_frequency))
        old = dict(zip(first["reference_roots"], first["candidate_roots"], strict=True))
        frequency = dict(zip(second["reference_roots"], second["candidate_roots"], strict=True))
        require(set(direct[name]) == set(old) and all(cross[name][direct[name][root]] == frequency[old[root]] for root in old),
                "The direct endpoint and cross-z assigned-root square is inconsistent.")
        paths[name] = [{"z4_one_ghz_root": root, "z6_one_ghz_root": old[root], "z4_8875_root": direct[name][root],
                        "z6_8875_via_z6": frequency[old[root]], "z6_8875_via_z4": cross[name][direct[name][root]]} for root in old]
    if published is not None:
        require(published == {"status": "consistent", "signed_paths": paths, "gauge_closure_required": False},
                "The production assigned-root square rows or no-holonomy policy differ.")
    return True


def replay_relation(safe_path, safe_sha, relation_path, relation_sha, runner_sha, comparison_sha):
    started = monotonic()
    result, safe, section, modes = replay_safe(safe_path, safe_sha, runner_sha)
    require(valid_sha(comparison_sha) and shared.file_sha(COMPARATOR) == comparison_sha,
            "The externally reviewed comparison producer SHA-256 differs.")
    relation = shared.read_receipt(relation_path, relation_sha)
    require(relation["content_sha256"] == digest({key: value for key, value in relation.items() if key != "content_sha256"})
            and relation["status"] == "signed_section_relation_accepted" and relation["phase"] == "saved_section_relation"
            and relation["frequency_hz"] == FREQUENCY and relation["reference_frequency_hz"] == 1_000_000_000
            and relation["positive_open_channels"] == relation["negative_open_channels"] == CHANNELS
            and relation["overlap_gate"] == .99 and same_artifact(relation["candidate_safe_receipt"], artifact(safe_path))
            and same_artifact(relation["candidate_safe_modes"], result["safe_modes"])
            and same_artifact(relation["producer"], artifact(COMPARATOR))
            and same_artifact(relation["reviewed_runner"], artifact(RUNNER))
            and relation["relative_wavenumber_gap"] == 1e-8 and relation["safe_eigensolve_performed"] is False
            and all(relation[key] is False for key in SCOPE),
            "The production relation receipt, reviewed producer or finite no-body scope differs.")
    expected_inputs = {}
    for role, (path, expected) in RELATION_PINS.items():
        require(shared.file_sha(path) == expected, f"A pinned relation input changed: {role}.")
        expected_inputs[role] = artifact(path)
    expected_inputs.update(candidate_safe_receipt=artifact(safe_path), candidate_safe_modes=result["safe_modes"])
    require(set(relation["inputs"]) == set(expected_inputs)
            and all(same_artifact(relation["inputs"][role], link) for role, link in expected_inputs.items()),
            "The production relation is bound to another scientific input or matching source.")
    require(shared.file_sha(Z6_SECTION) == Z6_SECTION_SHA and shared.file_sha(Z6_MODES) == Z6_MODES_SHA,
            "An accepted z6 section or frequency-mode artifact changed.")
    with np.load(Z6_SECTION, allow_pickle=False) as arrays:
        z6_section = {key: arrays[key] for key in ("section_vertices_yz_m", "section_triangles", "coordinates_yz_m", "component_dofs_xyz")}
    z6_modes = load_modes(Z6_MODES)
    current = signed_sections(section, modes, FREQUENCY)
    reference = signed_sections(section, section, 1_000_000_000, archived=True)
    z6 = signed_sections(z6_section, z6_modes, FREQUENCY)
    direct, direct_metrics = check_edge(relation["edges"]["z4_frequency"], reference, current)
    cross, cross_metrics = check_edge(relation["edges"]["cross_z_8875"], current, z6)
    previous = shared.read_receipt(OLD_THICKNESS, OLD_THICKNESS_SHA)
    old_frequency = shared.read_receipt(Z6_RELATION, Z6_RELATION_SHA)
    square = check_root_square(direct, cross, previous, old_frequency, relation["root_permutation_square"])
    result.update(relation_status="matched", relation=artifact(relation_path), assigned_root_square_consistent=square,
                  relation_replay={"status": "independently_verified_signed_relation", "z4_frequency": direct_metrics,
                      "cross_z_8875": cross_metrics, "assigned_root_square_consistent": square,
                      "gauge_policy": "Each direct edge checked independently; closed-loop phase holonomy is not an acceptance condition."})
    result["implementation"]["comparison"] = artifact(COMPARATOR)
    result["safe_replay_elapsed_seconds"] = result["elapsed_seconds"]
    result["elapsed_seconds"] = monotonic() - started
    return result


def check_prebody_verification(receipt, safe, safe_link, relation_link, receipt_path=None):
    require(receipt["status"] == "independently_verified_safe" and receipt["relation_status"] == "matched"
            and receipt["frequency_hz"] == FREQUENCY and receipt["problem_dofs"] == DOFS and receipt["port_dofs"] == PORT_DOFS
            and receipt["all_discrete_roots_replayed"] == 2 * PORT_DOFS
            and receipt["positive_open_channels"] == receipt["negative_open_channels"] == CHANNELS
            and receipt["incident_port_roots"] == safe["incident_port_roots"]
            and receipt["backward_port_roots"] == safe["backward_port_roots"]
            and same_artifact(receipt["safe_receipt"], safe_link) and same_artifact(receipt["safe_modes"], safe["safe_modes"])
            and same_artifact(receipt["relation"], relation_link) and receipt["assigned_root_square_consistent"] is True
            and all(receipt[key] is False for key in SCOPE),
            "The pre-body SAFE/relation replay is unbound, incomplete or outside finite scope.")
    producer, consumer = receipt["implementation"]["verifier"], artifact(Path(__file__))
    if same_artifact(producer, consumer):
        return {"status": "same_reviewed_producer", "accepted_producer": producer, "consumer": consumer}
    # Only the exact accepted receipt and retained 34e230... implementation may
    # cross this consumer-schema correction. Its numerical checks are not rerun.
    require(receipt_path is not None and shared.file_sha(Path(receipt_path)) == ACCEPTED_PREBODY_SHA
            and producer["sha256"] == ACCEPTED_PREBODY_PRODUCER_SHA
            and producer["bytes"] == 40611
            and producer["path"].replace("\\", "/").rsplit("/", 1)[-1] == "replay_lawrence_frequency_z4_AP_pnc.py"
            and shared.file_sha(ACCEPTED_PREBODY_PRODUCER) == ACCEPTED_PREBODY_PRODUCER_SHA
            and ACCEPTED_PREBODY_PRODUCER.stat().st_size == producer["bytes"],
            "The pre-body independent verifier source lacks its exact accepted receipt and retained producer.")
    return {"status": "verified_exact_saved_prebody_producer", "accepted_prebody_receipt": artifact(receipt_path),
            "accepted_producer": producer, "retained_producer": artifact(ACCEPTED_PREBODY_PRODUCER), "consumer": consumer,
            "scope": "Only receipt-consumer schema checks changed; accepted SAFE/relation evidence and all numerical gates are preserved."}


def check_phases(stage, safe, solved, exported, runner_sha):
    require(safe["status"] == "safe_ready" and stage["status"] == "operator_staged" and solved["status"] == "drives_accepted"
            and exported["status"] == "completed_unverified" and solved["stage"] == exported["stage"]
            and all(stage[key] == safe[key] for key in ("configuration", "safe_modes", "safe_section", "section_receipt",
                     "incident_port_roots", "backward_port_roots", "safe_section_array_sha256", "frequency_hz", "trace_sha256"))
            and all(phase["driver"]["sha256"] == runner_sha
                    and phase["implementation"] == safe["implementation"]
                    and phase["reviewed_sources"] == safe["reviewed_sources"]
                    and phase["superlu_library"] == safe["superlu_library"] for phase in (stage, solved, exported))
            and all(stage[key] is False for key in ("safe_eigensolve_performed", "phase_adjustment_applied",
                      "spatial_convergence_established", "frequency_convergence_established"))
            and all(exported[key] is False for key in ("spatial_convergence_established", "frequency_convergence_established"))
            and (("phase_adjustment_applied" in exported and exported["phase_adjustment_applied"] is False)
                 or ("phase_adjustment_applied" not in exported and runner_sha == FROZEN_RUNNER_SHA)),
            "The returned phases differ in reviewed sources, stage binding or finite scope.")
    report = solved["linear_solver"]
    count = report["factor"]["factor_nonzeros"]
    require(report["relative_target"] == shared.TARGET
            and report["relative_residual_limit"] == report["backward_residual_limit"] == shared.RESIDUAL
            and report["factor"]["permc_spec"] == "COLAMD" and type(count) is int and 2 * DOFS <= count <= np.iinfo(np.int32).max
            and all(report[key] == stage[key] for key in ("operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros"))
            and [row["incident_port_root"] for row in report["channels"]] == safe["incident_port_roots"],
            "The original gates, factor-index report or five accepted drive identities differ.")
    return report


def check_publication_phase(summary, stage, solved, exported):
    """Use explicit, linked public provenance; never infer a missing flag as False."""
    provenance = summary["checkpointed_scattering"]
    require(stage["phase_adjustment_applied"] is False and provenance["phase_adjustment_applied"] is False
            and provenance["reference_planes_m"] == stage["reference_planes_m"] == list(shared.PLANES)
            and provenance["prepared_frequency_stage"] == exported["stage"]
            and provenance["solve_receipt"] == exported["solve"]
            and all(provenance[name] == {key: phase[key] for key in EXECUTION_KEYS}
                    for name, phase in (("frequency_equation_preparation", stage), ("linux_solve", solved), ("linux_export", exported))),
            "The linked public phase flag, original planes or execution provenance differs.")
    return {"stage_phase_adjustment_applied": stage["phase_adjustment_applied"],
            "public_provenance_phase_adjustment_applied": provenance["phase_adjustment_applied"],
            "export_receipt_has_phase_flag": "phase_adjustment_applied" in exported,
            "reference_planes_m": provenance["reference_planes_m"],
            "basis": "Explicit False in the original stage and hash-linked public provenance, with exact stage/solve/export execution identities."}


def compare_public_tables(public, blocks, totals, incident, returning):
    """Join all 50 propagating coefficients and five rows at this exact frequency."""
    with (public / "scattering_channels_AP_pnc.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(kind, incoming, outgoing): (column, row)
                for kind, roots in (("reflection", returning), ("transmission", incident))
                for column, incoming in enumerate(incident) for row, outgoing in enumerate(roots)}
    require(len(rows) == len(expected) == 2 * CHANNELS**2, "The public table must retain all 50 propagating coefficients.")
    for row in rows:
        key = row["kind"], int(row["incident_port_index"]), int(row["outgoing_port_index"])
        require(key in expected, "A public coefficient is duplicated or has another signed root.")
        column, index = expected.pop(key)
        coefficient, power = blocks[key[0]]
        actual = complex(float(row["amplitude_real"]), float(row["amplitude_imag"]))
        require(int(row["incident_mode_order"]) == column + 1
                and np.isclose(actual, coefficient[index, column], rtol=1e-8, atol=1e-10)
                and np.isclose(float(row["power_fraction"]), power[index, column], rtol=1e-8, atol=1e-10),
                "A public complex coefficient or channel power differs from independent replay.")
    with (public / "scattering_AP_pnc.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == CHANNELS, "The public table must retain all five incident drives.")
    r, t, direct_r, direct_t, _ = totals
    metrics = {"reflection": r, "transmission": t, "reflection_flux": direct_r, "transmission_flux": direct_t,
               "power_balance": 1 - r - t, "flux_power_balance": 1 - direct_r - direct_t}
    for column, row in enumerate(rows):
        require(int(row["incident_mode_order"]) == column + 1 and float(row["frequency_GHz"]) == FREQUENCY / 1e9
                and all(np.isclose(float(row[key]), values[column], rtol=1e-8, atol=1e-10) for key, values in metrics.items()),
                "A public frequency or incident-drive power row differs.")


def replay_point(stage_path, safe_path, solve_path, export_path, export_sha, runner_sha, prebody_path):
    started = monotonic()
    checked_helpers()
    require(valid_sha(runner_sha) and shared.file_sha(RUNNER) == runner_sha, "The externally reviewed runner SHA-256 differs.")
    exported = shared.read_receipt(export_path, export_sha)
    stage = read(checked_link(stage_path, exported["stage"]))
    solved = read(checked_link(solve_path, exported["solve"]))
    safe = read(checked_link(safe_path, stage["safe_receipt"]))
    original = shared.read_receipt(ARCHIVE / "stages_AP_pnc.json", BODY_SHA)
    config = read(checked_link(ARCHIVE / "configuration_AP_pnc.json", original["configuration"]))
    check_archive_identity(stage, original, config, runner_sha)
    report = check_phases(stage, safe, solved, exported, runner_sha)
    public = export_path.parent / exported["public_summary"]["path"].replace("\\", "/").rsplit("/", 2)[-2]
    summary_path = checked_link(public / "summary_AP_pnc.json", exported["public_summary"])
    summary = read(summary_path)
    phase_evidence = check_publication_phase(summary, stage, solved, exported)
    prebody = read(checked_link(prebody_path, stage["safe_verification"]))
    compatibility = check_prebody_verification(prebody, safe, stage["safe_receipt"], stage["signed_relation"], prebody_path)
    incident, returning = safe["incident_port_roots"], safe["backward_port_roots"]
    core = frequency_core(incident, returning)
    arrays = stage["array_files"]
    body = {key: checked_link(ARCHIVE / f"{key}_AP_pnc.npz", arrays[key]) for key in ("K", "M")}
    paths = {key: checked_link(stage_path.parent / name, arrays[key]) for key, name in (("A", "A_AP_pnc.npz"), ("rhs", "rhs_AP_pnc.npy"))}
    modes_path = linked_basename(safe_path.parent, stage["safe_modes"])
    mesh_name = original["mesh"]["path"].replace("\\", "/")
    mesh = checked_link(ROOT / mesh_name, original["mesh"])
    require(summary["frequency_hz"] == FREQUENCY and summary["incident_side"] == "left" and summary["hole_count"] == 26
            and summary["ndof"] == DOFS and summary["port_dofs"] == PORT_DOFS and summary["num_incident_channels"] == CHANNELS
            and summary["incident_mode_indices"] == incident and summary["power_balance_passed"] is True
            and summary["spatial_convergence_established"] is False and summary["configuration"] == stage["configuration"]
            and all(summary["linear_solver"].get(key) == report[key] for key in ("relative_target", "relative_residual_limit",
                      "backward_residual_limit", "operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros")),
            "The public point differs from its verified five-drive scope or original equations.")
    for name in ("selected_fields_AP_pnc.npz", "selected_vertex_shapes_AP_pnc.vtu", "scattering_AP_pnc.csv",
                 "scattering_channels_AP_pnc.csv", "port_modes_AP_pnc.csv", "configuration_AP_pnc.json"):
        require(shared.file_sha(public / name) == summary["output_sha256"][name], f"Public output changed: {name}.")
    require(read(public / "configuration_AP_pnc.json") == stage["configuration"], "Public physical inputs differ.")
    available = psutil.virtual_memory().available
    required = max(arrays[key]["bytes"] for key in ("A", "K", "M")) + 3 * DOFS * CHANNELS * 16 + 512 * 2**20
    require(available >= required, f"Independent replay requires {required} B available; measured {available} B.")
    vectors = np.empty((DOFS, CHANNELS), dtype=np.complex128, order="F")
    for column, row in enumerate(report["channels"]):
        require(row["column"] == column and row["target_met"] is True and row["info"] == 0,
                "A drive lacks its accepted original-equation record.")
        vector = np.load(checked_link(solve_path.parent / f"accepted-{column:03d}_AP_pnc.npy", row["accepted_vector"]), allow_pickle=False)
        require(vector.shape == (DOFS,) and vector.dtype == np.complex128 and np.isfinite(vector).all(), "Invalid accepted displacement vector.")
        vectors[:, column] = vector
    operator = load_npz(paths["A"])
    rhs = np.load(paths["rhs"], allow_pickle=False, mmap_mode="r")
    require(operator.shape == (DOFS, DOFS) and operator.format == "csc" and operator.has_canonical_format
            and operator.dtype == np.complex128 and operator.indices.dtype == operator.indptr.dtype == np.int32
            and operator.nnz == stage["operator_nonzeros"] and np.isfinite(operator.data).all()
            and rhs.shape == vectors.shape and rhs.dtype == np.complex128 and np.isfinite(rhs).all()
            and shared.array_sha(operator.data, operator.indices, operator.indptr) == stage["operator_csc_sha256"]
            and shared.array_sha(rhs) == stage["rhs_array_sha256"], "The original complex128 CSC equation arrays differ.")
    relative, backward, rhs_norm, matrix_norm = core.check_original_equations(operator, rhs, vectors)
    del operator
    force = core.body_action(body["K"], vectors) - core.OMEGA**2 * core.body_action(body["M"], vectors)
    section = load_archived_section()
    with np.load(checked_link(ARCHIVE / "ports_AP_pnc.npz", original["port_modes"]), allow_pickle=False) as ports:
        section.update(left_dofs=ports["left_dofs"], right_dofs=ports["right_dofs"])
    modes = load_modes(modes_path)
    with np.load(public / "selected_fields_AP_pnc.npz", allow_pickle=False) as fields:
        core.replay_fields(fields, mesh, section, modes, vectors)
        blocks, totals, errors, rhs_error, boundary_error, coherent = core.replay_ports(
            modes, fields, section["left_dofs"], section["right_dofs"], vectors, force, rhs, rhs_norm)
    compare_public_tables(public, blocks, totals, incident, returning)
    r, t, direct_r, direct_t, net = totals
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "point_verified_within_saved_discretization",
            "frequency_hz": FREQUENCY, "problem_dofs": DOFS, "port_dofs": PORT_DOFS, "reference_planes_m": list(shared.PLANES),
            "public_summary": artifact(summary_path), "public_summary_sha256": exported["public_summary"]["sha256"],
            "phase_receipts": {key: artifact(value) for key, value in (("safe", safe_path), ("stage", stage_path), ("solve", solve_path), ("export", export_path))},
            "prebody_verification": artifact(prebody_path), "prebody_replay": artifact(prebody_path), "relation": stage["signed_relation"],
            "prebody_producer_compatibility": compatibility, "reference_plane_evidence": phase_evidence,
            "implementation": {"verifier": artifact(Path(__file__)), "shared_replay": artifact(Path(shared.__file__))},
            "original_equations_passed": True, "physical_nodal_force_power_passed": True, "all_modal_coefficients_passed": True,
            "full_p2_field_identity_passed": True, "coherent_left_incidence_passed": True,
            "propagating_coefficients_checked": 2 * CHANNELS**2, "full_directional_coefficients_checked": 2 * PORT_DOFS * CHANNELS,
            "maximum_relative_residual": float(max(relative)), "maximum_backward_residual": float(max(backward)),
            "matrix_infinity_norm": matrix_norm, "relative_target": core.TARGET, "relative_residual_limit": core.RESIDUAL,
            "backward_residual_limit": core.RESIDUAL, "power_limit": core.POWER,
            "maximum_power_errors": dict(zip(core.POWER_CHECKS, errors.tolist(), strict=True)), "coherent_left_incidence": coherent,
            "channels": [{"column": i, "incident_root": root, "reverse_root": returning[i], "relative_residual": relative[i],
                          "backward_residual": backward[i], "independent_rhs_relative_error": rhs_error[i],
                          "independent_body_boundary_relative_residual": boundary_error[i], "reflection": r[i], "transmission": t[i],
                          "reflection_flux": direct_r[i], "transmission_flux": direct_t[i], "net_outward_power_ratio": net[i]}
                         for i, root in enumerate(incident)],
            "elapsed_seconds": monotonic() - started, "available_memory_bytes_before_replay": available,
            "required_memory_bytes_before_replay": required, "spatial_convergence_established": False,
            "frequency_convergence_established": False, "full_two_port_unitarity_established": False, "reciprocity_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    relation = sub.add_parser("relation", help="One combined full-SAFE and direct signed-relation replay before body work.")
    for name in ("safe", "relation"):
        relation.add_argument(f"--{name}", type=Path, required=True)
        relation.add_argument(f"--{name}-sha256", required=True)
    relation.add_argument("--comparison-sha256", required=True)
    point = sub.add_parser("point")
    for name in ("safe", "stage", "solve", "export"):
        point.add_argument(f"--{name}", type=Path, required=True)
    point.add_argument("--export-sha256", required=True)
    point.add_argument("--prebody-replay", type=Path, required=True)
    for command in (relation, point):
        command.add_argument("--runner-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "relation":
        result = replay_relation(args.safe.resolve(), args.safe_sha256, args.relation.resolve(), args.relation_sha256,
                                 args.runner_sha256, args.comparison_sha256)
    else:
        result = replay_point(args.stage.resolve(), args.safe.resolve(), args.solve.resolve(), args.export.resolve(),
                              args.export_sha256, args.runner_sha256, args.prebody_replay.resolve())
    print(json.dumps(result, indent=2, allow_nan=False, default=lambda value: value.item() if isinstance(value, np.generic) else value))


if __name__ == "__main__":
    main()
