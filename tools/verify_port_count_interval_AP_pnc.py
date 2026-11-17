"""Independently audit a bounded SAFE count interval. Athul Prem | _AP_pnc.

No finite-element assembly, eigensolve, producer validator or branch matcher is
used. Original equations, reconstructed nodal forces and velocity-work flux are
replayed from immutable arrays. A finite count interval is not a unique cutoff.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = "verification/port_inventory_AP_pnc.json"
LEDGER_SHA256 = "d49f6a54848fc05a2bffdeda1141954a567e449ccbf2983b06c69533eae89a56"
ANCHOR_RECEIPTS = {
    100_000_000: ("point-100000000_AP_pnc.json", "c7d81677fbe2d67649e943c609562a6407e24a97e6ebb4b878583e39d8806344"),
    1_000_000_000: ("point-1000000000_AP_pnc.json", "38fac1f8bedda024e65bbf3fab654734654793b61a049d86cf30944879b76b40"),
}
INITIAL_INTERVAL = [100_000_000, 1_000_000_000]
TARGET_WIDTH_HZ = 56_250_000
MAX_NEW_POINTS = 4
RESIDUAL_LIMIT = 1e-8
TRACTION_LIMIT = 1e-8
FLUX_LIMIT_W = 1e-7
CONDITION_LIMIT = 1e10
PAIRING_LIMIT = 1e-8
MODE_FIELDS = ("frequency_hz", "wavenumbers_rad_m", "displacements", "axial_forces", "positive_indices",
               "negative_indices", "propagating", "powers_w", "relative_residuals")
RESOURCE_CONTRACT = {"numerical_threads": 1, "sample_period_seconds": 0.05,
    "process_tree_rss_limit_bytes": 1024 * 2**20, "available_reserve_bytes": 512 * 2**20,
    "launch_available_bytes": 1536 * 2**20, "point_wall_seconds": 90,
    "new_disk_limit_bytes": 512 * 2**20, "external_process_tree_supervisor_required": True}
SCOPE = ("Independent finite count-interval replay only. Root IDs are local to each frequency; "
         "no unique cutoff, monotonic channel count, continuous branch, device response or convergence is established.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def linked(link, base):
    base = Path(base).resolve()
    path = (base / link["path"]).resolve()
    require(path.is_relative_to(base) and path.is_file() and path.stat().st_size == link["bytes"]
            and sha256(path) == link["sha256"], f"Artifact identity differs: {link['path']}.")
    return path


def read_json(path):
    return json.loads(Path(path).read_bytes())


def pair_roots(waves, length_m):
    """Assign every root bijectively, including repeated-root multiplicities."""
    result = {}
    for name, target in (("conjugation", waves.conj()), ("reciprocity", -waves.conj())):
        distances = abs(target[:, None] - waves[None, :])
        rows, columns = linear_sum_assignment(distances)
        errors = distances[rows, columns]
        scaled = errors / np.maximum(1 / length_m, abs(target[rows]))
        require(len(rows) == len(waves) and np.isfinite(scaled).all()
                and float(scaled.max()) <= PAIRING_LIMIT, f"Independent {name} root pairing differs.")
        result[name] = {"maximum_mismatch_rad_m": float(errors.max()), "maximum_scaled_mismatch": float(scaled.max())}
    return result


def replay_modes(operators, arrays, frequency_hz, *, length_m, reference_speed_m_s, batch_size=32):
    """Replay all roots without the producer's polynomial/flux implementation."""
    require(type(batch_size) is int and batch_size > 0 and math.isfinite(length_m) and length_m > 0
            and math.isfinite(reference_speed_m_s) and reference_speed_m_s > 0, "Invalid independent replay settings.")
    matrices = {name: np.asarray(operators[name]) for name in ("K0", "M", "Q", "G")}
    n = len(matrices["M"])
    require(n > 0 and all(value.shape == (n, n) and value.dtype == np.float64 and np.isfinite(value).all()
                         for value in matrices.values()), "Invalid original SAFE operators.")
    require(set(MODE_FIELDS).issubset(arrays), "Incomplete saved root arrays.")
    omega = math.tau * frequency_hz
    require(math.isfinite(omega) and omega > 0 and float(arrays["frequency_hz"]) == frequency_hz,
            "Saved root frequency differs.")
    k, u, saved_force = (np.asarray(arrays[name]) for name in ("wavenumbers_rad_m", "displacements", "axial_forces"))
    opened = np.asarray(arrays["propagating"])
    saved_power, saved_residual = (np.asarray(arrays[name]) for name in ("powers_w", "relative_residuals"))
    directions = [np.asarray(arrays[name]) for name in ("positive_indices", "negative_indices")]
    require(u.shape == (n, 2*n) and saved_force.shape == u.shape and k.shape == (2*n,)
            and opened.shape == (2*n,) and opened.dtype == np.bool_
            and saved_power.shape == saved_residual.shape == (2*n,)
            and all(index.shape == (n,) and index.dtype.kind in "iu" for index in directions)
            and np.array_equal(np.sort(np.concatenate(directions)), np.arange(2*n)),
            "Invalid or incomplete signed root basis.")
    require(all(value.dtype.kind in "fciu" and np.isfinite(value).all()
                for value in (k, u, saved_force, saved_power, saved_residual))
            and np.all((saved_residual >= 0) & (saved_residual <= RESIDUAL_LIMIT))
            and np.all(k[opened].imag == 0) and np.all(k[~opened].imag != 0)
            and np.all(abs(k) * length_m > 8e-10), "Invalid finite root or cutoff metadata.")
    norms = np.linalg.norm(u, axis=0)
    require(np.all(norms > 0), "A saved root has zero displacement norm.")
    mass, stiffness, axial, coupling = (matrices[name] for name in ("M", "K0", "Q", "G"))
    dynamic = stiffness - omega**2 * mass
    dynamic_norm = np.linalg.norm(dynamic, ord=np.inf)
    skew_norm = np.linalg.norm(coupling.T - coupling, ord=np.inf)
    axial_norm = np.linalg.norm(axial, ord=np.inf)
    forces = np.empty(u.shape, dtype=np.complex128)
    residual, traction, power, relative_power = (np.empty(2*n) for _ in range(4))
    for start in range(0, 2*n, batch_size):
        selection = slice(start, min(2*n, start + batch_size))
        wave, displacement = k[selection], u[:, selection]
        qu, gu = axial @ displacement, coupling @ displacement
        force = gu + qu * (1j * wave)
        forces[:, selection] = force
        # This first-order equilibrium form differs from the producer's
        # expanded quadratic action and independently includes traction.
        defect = dynamic @ displacement + (coupling.T @ displacement - force) * (1j * wave)
        scale = (dynamic_norm + abs(wave) * skew_norm + abs(wave)**2 * axial_norm) * norms[selection]
        require(np.all(scale > 0), "Zero original-equation residual scale.")
        residual[selection] = np.linalg.norm(defect, axis=0) / scale
        force_scale = np.linalg.norm(gu, axis=0) + abs(wave) * np.linalg.norm(qu, axis=0)
        traction[selection] = np.linalg.norm(saved_force[:, selection] - force, axis=0) / np.maximum(force_scale, np.finfo(float).tiny)
        velocity = -1j * omega * displacement
        power[selection] = -0.5 * np.sum((velocity.conj() * force).real, axis=0)
        kinetic = 0.5 * omega**2 * np.sum((displacement.conj() * (mass @ displacement)).real, axis=0)
        require(np.all(kinetic > 0), "Nonpositive saved-mode kinetic normalization.")
        relative_power[selection] = abs(power[selection]) / (kinetic * reference_speed_m_s)
    require(np.isfinite(residual).all() and float(residual.max()) <= RESIDUAL_LIMIT,
            "Independent original eigenpair backward residual exceeds 1e-8.")
    require(np.isfinite(traction).all() and float(traction.max()) <= TRACTION_LIMIT,
            "Independent reconstructed traction differs by more than 1e-8.")
    power_difference = float(abs(power - saved_power).max())
    require(math.isfinite(power_difference) and power_difference <= FLUX_LIMIT_W,
            "Independent saved signed powers differ.")
    require(np.all(relative_power[opened] > 1e-8), "Independent open mode has unresolved vanishing power.")
    open_roots = np.flatnonzero(opened)
    _, group_ids, counts = np.unique(k[open_roots].real, return_inverse=True, return_counts=True)
    coincident = [open_roots[group_ids == index].tolist() for index in np.flatnonzero(counts > 1)]
    metrics = {"discrete_roots": 2*n, "maximum_eigen_backward_residual": float(residual.max()),
               "maximum_relative_traction_error": float(traction.max()), "maximum_modal_power_difference_w": power_difference,
               "maximum_saved_backward_residual_difference": float(abs(residual - saved_residual).max()),
               "minimum_open_relative_power": float(relative_power[opened].min()) if opened.any() else None,
               "root_partner_checks": pair_roots(k, length_m), "open_channels": [],
               "coincident_propagating_root_groups": coincident,
               "degeneracy_status": "coincident_propagating_roots" if coincident else "none_detected"}
    for name, sign, indices in zip(("positive", "negative"), (1, -1), directions, strict=True):
        propagating = opened[indices]
        require(np.all(sign * power[indices[propagating]] > 0)
                and np.all(sign * k[indices[~propagating]].imag > 0), "Independent signed direction differs.")
        displacement, force = u[:, indices], forces[:, indices]
        condition = float(np.linalg.cond(displacement / norms[indices]))
        velocity = -1j * omega * displacement
        work = velocity.conj().T @ force
        gram = -(work + work.conj().T) / 4
        flux_error = float(abs(gram - np.diag(sign * propagating.astype(float))).max())
        require(math.isfinite(condition) and condition < CONDITION_LIMIT and math.isfinite(flux_error)
                and flux_error <= FLUX_LIMIT_W, "Independent full directional flux or basis condition fails.")
        metrics[f"{name}_open_channels"] = int(propagating.sum())
        metrics[f"{name}_flux_gram_error_w"] = flux_error
        metrics[f"{name}_basis_condition"] = condition
        for root in indices[propagating]:
            metrics["open_channels"].append({"root_id": int(root), "direction": name,
                "k_real_rad_m": float(k[root].real), "k_imag_rad_m": float(k[root].imag),
                "signed_power_w": float(power[root]), "eigen_backward_residual": float(residual[root])})
    require(metrics["positive_open_channels"] == metrics["negative_open_channels"], "Independent open channel counts differ by direction.")
    return metrics


def compare_metrics(saved, replay):
    for name in ("positive_open_channels", "negative_open_channels", "discrete_roots"):
        require(saved[name] == replay[name], f"Published {name} differs from independent replay.")
    require(saved["coincident_propagating_root_groups"] == replay["coincident_propagating_root_groups"]
            and saved["degeneracy_status"] == replay["degeneracy_status"], "Published exact real-root degeneracy differs.")
    for name, limit in (("maximum_eigen_backward_residual", RESIDUAL_LIMIT),
                        ("maximum_relative_traction_error", TRACTION_LIMIT),
                        ("maximum_modal_power_difference_w", FLUX_LIMIT_W),
                        ("positive_flux_gram_error_w", FLUX_LIMIT_W), ("negative_flux_gram_error_w", FLUX_LIMIT_W)):
        require(type(saved[name]) in (float, int) and math.isfinite(saved[name]) and 0 <= saved[name] <= limit,
                f"Published acceptance metric {name} fails.")
    for name in ("positive_basis_condition", "negative_basis_condition"):
        require(math.isfinite(saved[name]) and 1 <= saved[name] < CONDITION_LIMIT, f"Published {name} fails.")
    require(len(saved["open_channels"]) == len(replay["open_channels"]), "Published signed row count differs.")
    for actual, expected in zip(saved["open_channels"], replay["open_channels"], strict=True):
        for name in ("root_id", "direction"):
            require(actual[name] == expected[name], f"Published signed {name} differs.")
        for name in ("k_real_rad_m", "k_imag_rad_m"):
            require(actual[name] == expected[name], f"Published signed {name} differs.")
        require(abs(actual["signed_power_w"] - expected["signed_power_w"]) <= FLUX_LIMIT_W
                and abs(actual["eigen_backward_residual"] - expected["eigen_backward_residual"]) <= RESIDUAL_LIMIT,
                "Published signed power or backward residual differs.")


def verify_history(identity, state, points):
    """Recompute each dyadic decision without assuming count monotonicity."""
    require(identity["schema_version"] == 1 and identity["initial_interval_hz"] == INITIAL_INTERVAL
            and identity["endpoint_counts"] == [4, 5] and identity["target_width_hz"] == TARGET_WIDTH_HZ
            and identity["max_new_points"] == MAX_NEW_POINTS, "Frozen interval contract differs.")
    require(identity["resource_contract"] == RESOURCE_CONTRACT, "Frozen resource contract differs.")
    require(state["identity_sha256"] == digest(identity) and len(points) <= MAX_NEW_POINTS
            and state["new_points_used"] == len(points), "Interval history identity or point budget differs.")
    lower, upper = INITIAL_INTERVAL
    unresolved = False
    for index, point in enumerate(points, 1):
        require(not unresolved and upper - lower > TARGET_WIDTH_HZ, "A point follows the milestone stop condition.")
        require(point["status"] in ("accepted", "failed", "interrupted"), "Unfinished or unknown point status cannot be verified.")
        require(point["index"] == index and point["frequency_hz"] == (lower + upper) // 2
                and point["interval_before_hz"] == [lower, upper]
                and point["identity_sha256"] == digest(identity), "Point is not the next exact dyadic midpoint.")
        payload = {key: value for key, value in point.items() if key != "point_sha256"}
        require(point["point_sha256"] == digest(payload), "Point receipt digest differs.")
        metrics = point.get("metrics")
        resolved = point["status"] == "accepted" and metrics is not None
        require(point["actual_environment"] == "inputs_AP_pnc.json:identity.archive.current_implementation",
                "Point actual-environment provenance differs.")
        if point["status"] == "accepted":
            resources, admission = point["resources"], point["admission"]
            require(point["eigensolve_performed"] is True and not resources["guard_observations"]
                    and 0 <= resources["peak_sampled_process_tree_rss_bytes"] <= RESOURCE_CONTRACT["process_tree_rss_limit_bytes"]
                    and resources["minimum_available_bytes"] >= RESOURCE_CONTRACT["available_reserve_bytes"]
                    and 0 <= resources["elapsed_seconds"] <= RESOURCE_CONTRACT["point_wall_seconds"]
                    and resources["sampling_interval_seconds"] == RESOURCE_CONTRACT["sample_period_seconds"]
                    and admission["available_bytes"] >= RESOURCE_CONTRACT["launch_available_bytes"]
                    and admission["free_disk_bytes"] >= admission["required_new_disk_bytes"] == 64 * 2**20
                    and admission["existing_output_bytes"] + admission["required_new_disk_bytes"] <= RESOURCE_CONTRACT["new_disk_limit_bytes"],
                    "Accepted point exceeded its recorded resource contract.")
        if resolved:
            counts = [metrics["positive_open_channels"], metrics["negative_open_channels"]]
            resolved = (counts in ([4, 4], [5, 5]) and metrics.get("degeneracy_status") == "none_detected"
                        and metrics.get("cutoff_status") == "solver_resolved_at_sample"
                        and not metrics.get("coincident_propagating_root_groups"))
        if resolved and counts == [4, 4]:
            decision, lower = "replace_lower", point["frequency_hz"]
        elif resolved and counts == [5, 5]:
            decision, upper = "replace_upper", point["frequency_hz"]
        else:
            decision, unresolved = "stop_unresolved", True
        require(point["decision"] == decision and point["interval_after_hz"] == [lower, upper],
                "Published dyadic count decision or resulting interval differs.")
    require(state["lower_hz"] == lower and state["upper_hz"] == upper and state["width_hz"] == upper - lower
            and state["target_width_hz"] == TARGET_WIDTH_HZ
            and state["cross_frequency_branch_identity"] == "unresolved", "Published final interval or branch status differs from its receipts.")
    # The file may describe a saved partial workflow. It must never label that
    # partial history as the target-width result.
    complete = not unresolved and upper - lower <= TARGET_WIDTH_HZ
    expected_status = "complete" if complete else "unresolved" if unresolved else "pending"
    require(state["status"] == expected_status, "Published status differs from the bounded stop condition.")
    if complete:
        reason = "target_sampled_width_reached"
    elif unresolved:
        last = points[-1]
        if last["status"] == "accepted":
            reason = "root_degeneracy" if last["metrics"]["degeneracy_status"] != "none_detected" or last["metrics"]["coincident_propagating_root_groups"] else "unexpected_directional_count"
        else:
            require(bool(last.get("error")) or last.get("stop_reason") == "previous_worker_interrupted",
                    "Failed/interrupted receipt has no retained reason.")
            reason = last["stop_reason"]
    else:
        reason = "point_budget_pause"
    require(state["stop_reason"] == reason, "Published stop reason differs from the immutable decision history.")
    return {"lower_hz": lower, "upper_hz": upper, "width_hz": upper - lower,
            "new_points_used": len(points), "target_width_reached": complete, "stopped_unresolved": unresolved}


def audit_archived_inputs(identity, root=ROOT):
    """Reuse pinned endpoint evidence; do not replay or solve old mode fields."""
    root = Path(root).resolve()
    archive = identity["archive"]
    ledger_path = linked(archive["inventory_ledger"], root)
    require(ledger_path == (root / LEDGER_PATH).resolve() and sha256(ledger_path) == LEDGER_SHA256,
            "Accepted inventory ledger identity differs.")
    ledger = read_json(ledger_path)
    safe_path = linked(archive["safe_receipt"], root)
    section_path = linked(archive["section_modes"], root)
    inputs_path = linked(archive["inventory_inputs"], root)
    require(archive["safe_receipt"]["sha256"] == ledger["archived_section"]["receipt"]["sha256"]
            and safe_path == (root / ledger["archived_section"]["receipt"]["path"]).resolve()
            and archive["section_modes"]["sha256"] == ledger["archived_section"]["mode_archive"]["sha256"]
            and section_path == (root / ledger["archived_section"]["mode_archive"]["path"]).resolve()
            and archive["inventory_inputs"]["sha256"] == ledger["publication"]["inputs"]["sha256"]
            and inputs_path == (root / ledger["publication"]["inputs"]["path"]).resolve(),
            "Frozen original section or inventory binding differs.")
    safe, inventory_inputs = read_json(safe_path), read_json(inputs_path)
    require(safe["ports"]["sha256"] == archive["section_modes"]["sha256"]
            and safe["status"] == "safe_ready" and safe["trace_dofs"] == 975
            and inventory_inputs["identity_sha256"] == digest(inventory_inputs["identity"])
            == ledger["publication"]["inventory_identity_sha256"], "Original accepted receipt identity differs.")
    require(archive["original_implementation"] == safe["implementation"], "Original implementation provenance differs.")
    anchors = archive["anchors"]
    require(len(anchors) == 2, "Exactly two accepted endpoints must be reused.")
    prior = {point["frequency_hz"]: point for point in ledger["independent_windows_replay"]["points"]}
    for item, frequency, count in zip(anchors, INITIAL_INTERVAL, (4, 5), strict=True):
        require(item["frequency_hz"] == frequency and item["count"] == count,
                "Accepted endpoint frequency or count differs.")
        receipt_path = linked(item["receipt"], root)
        modes_path = linked(item["modes"], root)
        accepted = prior[frequency]
        require(item["modes"]["sha256"] == accepted["artifact_sha256"]
                and modes_path == (root / accepted["root_artifact"]).resolve()
                and accepted["positive_open_channels"] == accepted["negative_open_channels"] == count,
                "Endpoint differs from accepted independent replay.")
        receipt = read_json(receipt_path)
        original_payload = {key: value for key, value in receipt.items() if key != "point_sha256"}
        filename, frozen_hash = ANCHOR_RECEIPTS[frequency]
        require(receipt_path == inputs_path.parent / filename and sha256(receipt_path) == frozen_hash
                and receipt["frequency_hz"] == frequency
                and receipt["identity_sha256"] == inventory_inputs["identity_sha256"]
                and receipt["point_sha256"] == digest(original_payload)
                and receipt["status"] == ("accepted" if frequency == INITIAL_INTERVAL[0] else "archived")
                and receipt["positive_open_channels"] == receipt["negative_open_channels"] == count,
                "Endpoint receipt binding differs.")
        if frequency == INITIAL_INTERVAL[0]:
            require(receipt["modes"]["sha256"] == item["modes"]["sha256"], "Low endpoint mode binding differs.")
        else:
            require(modes_path == section_path and item["modes"]["sha256"] == safe["ports"]["sha256"]
                    and receipt.get("eigensolve_performed") is False, "Archived high endpoint section binding differs.")
    current = archive["current_implementation"]
    require(current["source_sha256"] == safe["implementation"]["source_sha256"]
            and isinstance(current["packages"], dict) and isinstance(current["python"], str)
            and isinstance(current["platform"], str)
            and archive["python_version_changed"] == (current["python"] != archive["original_implementation"]["python"]),
            "Numerical source or separate environment provenance differs.")
    for name, expected in current["source_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file() and sha256(path) == expected,
                f"Frozen numerical source differs: {name}.")
    producer = linked(identity["producer"], root)
    require(producer == (root / "tools/refine_port_count_interval_AP_pnc.py").resolve(), "Producer role path differs.")
    return section_path


def verify_csv(directory, points):
    columns = ("frequency_hz", "status", "root_id", "direction", "k_real_rad_m", "k_imag_rad_m",
               "signed_power_w", "eigen_backward_residual", "receipt", "point_sha256", "error")
    with (Path(directory) / "port_count_interval_AP_pnc.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(columns), "Count interval CSV columns differ.")
        actual = list(reader)
    expected = []
    for point in points:
        for channel in point.get("metrics", {}).get("open_channels", []) or [{}]:
            row = {**channel, "frequency_hz": point["frequency_hz"], "status": point["status"],
                   "receipt": f"point-{point['frequency_hz']}_AP_pnc.json",
                   "point_sha256": point["point_sha256"], "error": point.get("error", "")}
            expected.append({name: str(row[name]) if name in row else "" for name in columns})
    require(actual == expected, "Count interval CSV rows differ from immutable receipts.")
    return len(actual)


def load_original_operators(path):
    require(Path(path).stat().st_size <= 64 * 1024**2, "Independent verifier refuses oversized section arrays.")
    with np.load(path, allow_pickle=False) as values:
        operators = {name: values[name] for name in ("K0", "M", "Q", "G")}
        coordinates = values["coordinates_yz_m"]
        length = float(np.ptp(coordinates, axis=0).max())
        speed = float(values["reference_speed_m_s"])
    require(operators["M"].shape == (975, 975) and coordinates.shape == (975, 2), "Frozen section dimension differs.")
    return operators, length, speed


def verify_interval(directory, *, root=ROOT):
    directory = Path(directory).resolve()
    inputs = read_json(directory / "inputs_AP_pnc.json")
    identity = inputs["identity"]
    require(inputs["identity_sha256"] == digest(identity), "Frozen interval inputs digest differs.")
    section_path = audit_archived_inputs(identity, root)
    state = read_json(directory / "interval_state_AP_pnc.json")
    points = []
    for link in state["points"]:
        path = linked(link, directory)
        point = read_json(path)
        require(path == directory / f"point-{point['frequency_hz']}_AP_pnc.json", "Point receipt role path differs.")
        points.append(point)
    history = verify_history(identity, state, points)
    require(sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
            <= RESOURCE_CONTRACT["new_disk_limit_bytes"], "Published interval exceeds its disk ceiling.")
    operators, length, speed = load_original_operators(section_path)
    replays = []
    for point in points:
        if point.get("modes") is None:
            require(point["decision"] == "stop_unresolved", "Resolved point has no retained root arrays.")
            continue
        modes_path = linked(point["modes"], directory)
        require(modes_path == directory / f"modes-{point['frequency_hz']}_AP_pnc.npz", "New mode role path differs.")
        require(modes_path.stat().st_size <= 64 * 1024**2, "Independent verifier refuses oversized root arrays.")
        with np.load(modes_path, allow_pickle=False) as values:
            arrays = {name: values[name] for name in MODE_FIELDS}
        replay = replay_modes(operators, arrays, point["frequency_hz"], length_m=length, reference_speed_m_s=speed)
        compare_metrics(point["metrics"], replay)
        replays.append({"frequency_hz": point["frequency_hz"], "point_sha256": point["point_sha256"], **replay})
    csv_rows = verify_csv(directory, points)
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "scope": SCOPE,
            "status": "verified_finite_count_interval" if history["target_width_reached"] else "verified_partial_or_unresolved_history",
            "identity_sha256": inputs["identity_sha256"], "history": history, "points": replays,
            "checked_new_roots": sum(item["discrete_roots"] for item in replays), "checked_csv_rows": csv_rows,
            "replayed_archived_roots": 0, "eigensolve_performed": False, "assembly_performed": False,
            "cross_frequency_branch_identity_established": False, "unique_cutoff_established": False,
            "monotonic_channel_count_established": False,
            "limits": ["Saved MODE_FIELDS omit left eigenvectors; the producer's uncertainty-based cutoff/pairing rejection cannot be reconstructed here.",
                       "Independent finite nonzero roots, signed decay/power, reciprocal pairing and original-equation gates are checked.",
                       "Unexamined intervals and any unresolved points remain unresolved; no device frequency is automatically selected."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "Independent verification output must be a fresh path.")
    result = verify_interval(arguments.directory)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checked_new_roots", "checked_csv_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()

