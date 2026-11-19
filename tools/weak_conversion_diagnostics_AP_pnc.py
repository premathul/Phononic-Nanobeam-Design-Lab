"""Diagnose saved 887.5 MHz reflected conversion at the original end planes.

Athul Prem | _AP_pnc
"""

from __future__ import annotations

from cmath import isfinite, phase
from hashlib import sha256
import json
from math import degrees, isclose
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "configs/saved_device_dataset_AP_pnc.json"
INPLANE_SOURCE = "data/saved_device_AP_pnc/controls/frequency_inplane_8875_AP_pnc.json"
THICKNESS_SOURCE = "data/saved_device_AP_pnc/controls/frequency_thickness_8875_AP_pnc.json"
FREQUENCY_HZ = 887_500_000


def reflection_cross_ratio(r11: complex, r12: complex, r21: complex, r22: complex) -> complex:
    """Return R12 R21 / (R11 R22), with outgoing rows and incident columns.

    Indices 1 and 2 of this block denote matched modes 1 and 5, respectively.
    Independent nonzero complex row and column factors cancel in the ratio.
    """
    if r11 == 0 or r22 == 0:
        raise ValueError("The reflection cross-ratio requires two nonzero diagonal amplitudes")
    return (r12 * r21) / (r11 * r22)


def _read_control(root: Path, source: str, expected_hash: str) -> dict:
    raw = (root / source).read_bytes()
    if sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"Saved control differs from the dataset manifest: {source}")
    control = json.loads(raw)
    if control["frequency_hz"] != FREQUENCY_HZ or control["phase_adjustment_applied"] is not False:
        raise ValueError(f"Expected native 887.5 MHz amplitudes at the original planes: {source}")
    return control


def _response_diagnostics(control: dict, response_key: str, selected_prefix: str) -> dict:
    response = control["raw_responses"][response_key]
    mode_1_to_5, mode_5_to_1 = control["selected_pair"]
    mode_1_in = mode_1_to_5[f"{selected_prefix}incident_root"]
    mode_5_out = mode_1_to_5[f"{selected_prefix}outgoing_root"]
    mode_5_in = mode_5_to_1[f"{selected_prefix}incident_root"]
    mode_1_out = mode_5_to_1[f"{selected_prefix}outgoing_root"]
    reverse_roots = {row["root"]: row["reverse_root"] for row in response["rows"]}
    if (response["frequency_hz"] != FREQUENCY_HZ or mode_1_in == mode_5_in or mode_1_out == mode_5_out
            or reverse_roots.get(mode_1_in) != mode_1_out or reverse_roots.get(mode_5_in) != mode_5_out):
        raise ValueError(f"Inconsistent native reciprocal roots: {response_key}")

    # Matrix positions are (outgoing root, incident root).
    roots = {"r11": (mode_1_out, mode_1_in), "r12": (mode_1_out, mode_5_in),
             "r21": (mode_5_out, mode_1_in), "r22": (mode_5_out, mode_5_in)}
    reflection = {}
    for coefficient in response["coefficients"]:
        pair = coefficient["outgoing_root"], coefficient["incident_root"]
        if coefficient["kind"] == "reflection" and pair in roots.values():
            if pair in reflection:
                raise ValueError(f"Duplicate reflection coefficient for {pair}: {response_key}")
            reflection[pair] = coefficient

    amplitudes, powers, components = {}, {}, {}
    for name, pair in roots.items():
        if pair not in reflection:
            raise ValueError(f"Missing reflection coefficient for {pair}: {response_key}")
        coefficient = reflection[pair]
        amplitude = complex(*coefficient["amplitude"])
        power = coefficient["power"]
        if not isfinite(amplitude) or not isclose(power, abs(amplitude) ** 2, rel_tol=1e-12):
            raise ValueError(f"Inconsistent reflection amplitude and power for {pair}: {response_key}")
        amplitudes[name] = amplitude
        powers[name] = power
        components.update({f"{name}_real": amplitude.real, f"{name}_imag": amplitude.imag})

    power_1_to_5, power_5_to_1 = powers["r21"], powers["r12"]
    mean_power = (power_1_to_5 + power_5_to_1) / 2
    if mean_power <= 0:
        raise ValueError(f"Relative reciprocity discrepancy requires a positive mean power: {response_key}")
    ratio = reflection_cross_ratio(**amplitudes)
    return {
        "mode_1_incident_root": mode_1_in,
        "mode_1_reflected_root": mode_1_out,
        "mode_5_incident_root": mode_5_in,
        "mode_5_reflected_root": mode_5_out,
        "power_1_to_5": power_1_to_5,
        "power_5_to_1": power_5_to_1,
        "mean_power": mean_power,
        "reciprocity_relative_difference": abs(power_1_to_5 - power_5_to_1) / mean_power,
        "cross_ratio_real": ratio.real,
        "cross_ratio_imag": ratio.imag,
        "cross_ratio_abs": abs(ratio),
        "cross_ratio_phase_deg": degrees(phase(ratio)),
        **components,
        "run_hash": response["run_hash"],
    }


def load_diagnostics(root: Path | None = None) -> list[dict]:
    """Read the pinned controls and return one flat diagnostic record per mesh."""
    root = ROOT if root is None else root
    manifest_raw = (root / MANIFEST).read_bytes()
    manifest_hash = sha256(manifest_raw).hexdigest()
    hashes = {entry["source"]["path"]: entry["source"]["sha256"]
              for entry in json.loads(manifest_raw)["controls"]}
    inplane = _read_control(root, INPLANE_SOURCE, hashes[INPLANE_SOURCE])
    thickness = _read_control(root, THICKNESS_SOURCE, hashes[THICKNESS_SOURCE])
    if (inplane["raw_responses"]["candidate_h80"] != thickness["raw_responses"]["reference"]
            or inplane["inputs"]["h80_8875_coefficients"] != thickness["inputs"]["z4_8875_coefficients"]):
        raise ValueError("The 887.5 MHz controls disagree on their shared h80 z4 response")

    meshes = (
        (100, 4, inplane, INPLANE_SOURCE, "reference_h100", "h100", ""),
        (80, 4, inplane, INPLANE_SOURCE, "candidate_h80", "h80", ""),
        (80, 6, thickness, THICKNESS_SOURCE, "candidate", "z6", "candidate_"),
    )
    rows = []
    for h_nm, layers, control, source, response_key, input_resolution, selected_prefix in meshes:
        native_source = control["inputs"][f"{input_resolution}_8875_coefficients"]
        rows.append({
            "mesh": f"h{h_nm} z{layers}",
            "global_inplane_target_nm": h_nm,
            "quadratic_thickness_layers": layers,
            "physical_thickness_nm": 200,
            "frequency_hz": FREQUENCY_HZ,
            **_response_diagnostics(control, response_key, selected_prefix),
            "portable_control_source": source,
            "portable_control_sha256": hashes[source],
            "native_887500000_hz_coefficient_archive_source": native_source["path"],
            "native_887500000_hz_coefficient_archive_sha256": native_source["sha256"],
            "dataset_manifest_source": MANIFEST,
            "dataset_manifest_sha256": manifest_hash,
        })
    return rows
