"""Replay saved native-map flux without assembling or solving the body. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np
from scipy.linalg import solve

from tools import map_scattering_AP_pnc as map_tool
from tools import replay_lawrence_device_point_AP_pnc as replay


def replay_saved_point(directory, frequency_ghz):
    directory = directory.resolve()
    state = map_tool.read_map_checkpoint(directory)
    published = map_tool.read_saved_map(directory)
    checkpoint_sha = replay.file_sha(directory / map_tool.CHECKPOINT)
    replay.require(published["checkpoint_sha256"] == checkpoint_sha,
                   "The published map and checkpoint differ.")
    point, = (p for p in state["points"] if p["frequency_GHz"] == frequency_ghz and p["status"] == "accepted")
    shared = state["shared"]
    point_dir, summary = map_tool._read_accepted(point, state["inputs"]["configuration"], shared, directory)
    replay.require(summary["incident_side"] == "left" and summary["configuration"]["mesh"]["element_order"] == 2,
                   "Expected a left-incident P2 point.")
    frequency_hz = summary["frequency_hz"]
    replay.OMEGA = 2 * np.pi * frequency_hz
    dofs = summary["ndof"]
    links = shared["artifacts"]
    stiffness, mass, section = (map_tool._verified(links[key], directory)
                                for key in ("stiffness", "mass", "port_section"))
    with np.load(section, allow_pickle=False) as arrays:
        left, right = arrays["left_dofs"], arrays["right_dofs"]
    with (point_dir / "port_modes_AP_pnc.csv").open(newline="", encoding="utf-8") as stream:
        roots = list(csv.DictReader(stream))
    with np.load(point_dir / map_tool.FIELDS, allow_pickle=False) as fields:
        vectors = fields["displacements_per_sqrt_w"]
        incident = fields["incident_mode_indices"]
        returning = fields["reflection_mode_indices"]
        replay.require(vectors.shape == (dofs, len(incident)) and np.isfinite(vectors).all()
                       and np.all(fields["frequencies_hz"] == frequency_hz)
                       and np.array_equal(incident, summary["incident_mode_indices"])
                       and np.array_equal(incident, fields["transmission_mode_indices"]),
                       "Saved P2 displacement columns and frequency-local roots differ.")
        coordinates = fields["dof_coordinates_m"][:, 0]
        for side, trace, plane in zip(("left", "right"), (left, right), shared["reference_planes_x_m"], strict=True):
            replay.require(np.array_equal(trace, fields[f"port_{side}_dofs"])
                           and np.array_equal(np.sort(trace), np.flatnonzero(np.isclose(coordinates, plane, rtol=0, atol=2e-20))),
                           f"The ordered {side} trace differs from its original port plane.")
        wave = fields["port_wavenumbers_rad_m"]
        replay.require(np.array_equal(wave, [complex(float(row["k_real_rad_m"]), float(row["k_imag_rad_m"])) for row in roots])
                       and [int(row["port_index"]) for row in roots] == list(range(len(wave))),
                       "Published SAFE root numbers or wavenumbers differ from the field export.")
        propagating = np.asarray([row["propagating"] == "True" for row in roots])
        plus, minus = (fields[f"{label}_all_mode_indices"] for label in ("transmission", "reflection"))
        for direction, indices in (("+x", plus), ("-x", minus)):
            replay.require(np.array_equal(np.sort(indices), [int(row["port_index"]) for row in roots if row["direction"] == direction]),
                           f"The saved {direction} basis does not contain every root in that direction.")
        u, f = fields["port_displacements"], fields["port_axial_forces"]
        modes = {"positive_indices": plus, "negative_indices": minus, "propagating": propagating,
                 "wavenumbers_rad_m": wave, "displacements": u, "axial_forces": f}
        nodal_force = replay.body_action(stiffness, vectors, expected_dofs=dofs)
        nodal_force -= replay.OMEGA**2 * replay.body_action(mass, vectors, expected_dofs=dofs)
        rhs = np.zeros_like(vectors, order="C")
        incident_u = u[:, incident]
        z_left = solve(u[:, minus].T, -f[:, minus].T).T
        rhs[left] = -f[:, incident] - z_left @ incident_u
        _, totals, power_errors, rhs_error, boundary_error, coherent = replay.replay_ports(
            modes, fields, left, right, vectors, nodal_force, rhs, np.linalg.norm(rhs, axis=0),
            incident_roots=incident.tolist(), returning_roots=returning.tolist())

    equations = summary["checkpointed_scattering"]["original_equations"]
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "passed_saved_discretization_flux_replay",
            "frequency_hz": frequency_hz, "reference_planes_x_m": shared["reference_planes_x_m"],
            "point_summary": point["accepted"]["summary"], "map_checkpoint_sha256": checkpoint_sha,
            "map_source_git_commit": state["source_git_commit"],
            "replay_source_sha256": replay.file_sha(Path(replay.__file__)),
            "regional_replay_source_sha256": replay.file_sha(Path(__file__)),
            "stiffness_mass_and_port_section": {key: links[key] for key in ("stiffness", "mass", "port_section")},
            "fields_sha256": summary["output_sha256"][map_tool.FIELDS], "body_dofs": dofs,
            "port_dofs": len(left), "original_equations_saved": equations,
            "maximum_relative_port_residual_saved": summary["maximum_relative_port_residual"],
            "original_rhs_matches_saved_sha256": map_tool._array_digest(rhs) == equations["rhs_sha256"],
            "physical_boundary_force_relative_residual_limit": replay.RESIDUAL,
            "physical_boundary_force_relative_residuals": boundary_error.tolist(),
            "rhs_trace_reconstruction_relative_differences": rhs_error.tolist(),
            "single_drive_power_limit": replay.POWER,
            "single_drive_power_error_maxima": dict(zip(replay.POWER_CHECKS, power_errors.tolist(), strict=True)),
            "single_drive_powers_in_incident_root_order": dict(zip(
                ("modal_reflection", "modal_transmission", "direct_reflection", "direct_transmission", "net_outward_power_ratio"),
                (values.tolist() for values in totals), strict=True)),
            "coherent_left_incidence": coherent}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_directory", type=Path)
    parser.add_argument("frequency_GHz", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with map_tool.workflow.measured_resources() as resources:
        result = replay_saved_point(args.map_directory, args.frequency_GHz)
    result["replay_resources"] = resources
    with args.output.open("x", encoding="utf-8") as report:
        json.dump(result, report, indent=2, allow_nan=False)
        report.write("\n")
