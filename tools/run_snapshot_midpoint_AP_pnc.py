"""Predict one archived thin-device midpoint from its two endpoint fields. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from src.phononic.workflow_AP_pnc import measured_resources, write_json
from tools import snapshot_grid_inputs_AP_pnc as archive
from tools import snapshot_native_system_AP_pnc as native
from tools import snapshot_response_AP_pnc as projection
from tools.read_thin_device_grid_AP_pnc import COMPARISON, LIMIT, TARGETS


EQUATION_LIMIT = 1e-9
RANK_RTOL = 1e-12


def _as_pairs(block):
    return np.stack((block.real, block.imag), axis=-1).tolist()


def _midpoint_rows(level, system, reduced, predicted):
    point = level.midpoint
    expected = point.reflection, point.transmission
    powers = system.modal_power_fractions(*predicted)
    reference_powers = system.modal_power_fractions(*expected)
    names = ("reflection", "transmission")
    rows = []
    for col, drive in enumerate(point.reference_drives):
        row = {"reference_drive": int(drive), "native_incident_root": int(point.incident_roots[col]),
               "relative_equation_residual": float(reduced.relative_residuals[col]),
               "minimum_relative_equation_residual_in_trial_span": float(reduced.minimum_relative_residuals[col])}
        for k, name in enumerate(names):
            candidate = float(np.sum(powers[k][:, col]))
            reference = float(np.sum(reference_powers[k][:, col]))
            row[name] = {"candidate_power_fraction": candidate, "direct_power_fraction": reference,
                         "power_error_percentage_points": 100 * (candidate - reference),
                         "maximum_absolute_complex_error": float(np.max(abs(predicted[k][:, col] - expected[k][:, col])))}
        row["power_budget_defect"] = 1 - sum(row[q]["candidate_power_fraction"] for q in names)
        row["total_power_comparison_passed"] = all(abs(row[q]["power_error_percentage_points"]) < 100 * LIMIT for q in names)
        row["complex_comparison_passed"] = all(row[q]["maximum_absolute_complex_error"] < LIMIT for q in names)
        row["relative_equation_check_passed"] = row["relative_equation_residual"] <= EQUATION_LIMIT
        rows.append(row)
    return rows


def _direct_replay(point, system, rhs):
    # Open the held-out volume solution only after the reduced predictions have been fixed.
    with np.load(point.fields_path, allow_pickle=False) as fields:
        direct = fields["displacements_per_sqrt_w"]
    residual = float(np.max(system.relative_residuals(direct, rhs)))
    reflected, transmitted = system.outgoing(direct)
    error = float(max(np.max(abs(reflected - point.reflection)), np.max(abs(transmitted - point.transmission))))
    if not (residual <= EQUATION_LIMIT and error < 1e-9):
        raise ValueError("Midpoint equation or port replay failed on its archived direct solution.")
    return {"maximum_relative_equation_residual": residual, "maximum_absolute_native_coefficient_difference": error}


def predict(mesh_nm):
    level = archive.load_snapshot_grid(mesh_nm)
    point = level.midpoint
    volume = native.VolumeOperators(*(load_npz(level.preparation[q]) for q in ("stiffness", "mass")))
    system = native.NativeScatteringSystem.from_saved_fields(volume, point.frequency_hz, point.port_fields)
    for group in ("incident_roots", "reflection_roots", "transmission_roots"):
        if not np.array_equal(getattr(point, group), getattr(system, group)):
            raise ValueError(f"Archive and midpoint system differ in native {group}.")
    trial = np.column_stack([end.displacements for end in level.endpoints])
    rhs = system.forcing()
    reduced = projection.solve_snapshot_response(trial, system.apply, rhs, rank_rtol=RANK_RTOL)
    predicted = system.outgoing(reduced.solution)
    rows = _midpoint_rows(level, system, reduced, predicted)
    return {
        "author": "Athul Prem", "file_tag": "_AP_pnc", "central_max_edge_nm": mesh_nm,
        "body_dofs": level.body_dofs, "training_frequencies_hz": [end.frequency_hz for end in level.endpoints],
        "prediction_frequency_hz": point.frequency_hz, "reference_planes_x_m": [0.0, 1.665e-5],
        "method": "Complex Galerkin projection of the midpoint body and outgoing traction operator onto all five left-incident volume responses at each endpoint. The midpoint ports and forcing retain their native phase. Only endpoint volume fields enter the trial basis.",
        "rank_relative_singular_value_cutoff": RANK_RTOL, "trial_columns": trial.shape[1],
        "trial_rank": reduced.rank, "applied_trial_rank": reduced.action_rank,
        "projected_operator_condition_2": reduced.condition,
        "relative_equation_residual_limit": EQUATION_LIMIT, "output_comparison_limit": LIMIT,
        "drives": rows,
        "native_coefficients": {"incident_roots_by_column": list(map(int, point.incident_roots)),
                                "reflection_roots_by_row": list(map(int, point.reflection_roots)),
                                "transmission_roots_by_row": list(map(int, point.transmission_roots)),
                                "format": "Each entry is [real, imaginary].",
                                "reflection": _as_pairs(predicted[0]), "transmission": _as_pairs(predicted[1])},
        "held_out_direct_solution_replay": _direct_replay(point, system, rhs),
        "scope": "One midpoint on one finite mesh. Output discrepancies are against its already accepted direct result. The minimum relative residual is the independent least-squares fit for each forcing in the retained span. This comparison does not establish mesh or frequency convergence.",
        "inputs": {"grid_comparison": COMPARISON, "level": level.evidence,
                   "endpoints": [end.evidence for end in level.endpoints], "midpoint": point.evidence},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-nm", type=float, choices=TARGETS, required=True)
    parser.add_argument("--output", type=Path, help="Write a new JSON file outside the native archive.")
    args = parser.parse_args(argv)
    if args.output is not None:
        output = args.output.resolve()
        if output.suffix != ".json" or output.exists() or output.is_relative_to((ROOT / "results/runs").resolve()):
            parser.error("--output must be a new JSON file outside the native archive")
    with measured_resources() as resources:
        result = predict(args.mesh_nm)
    result["resources"] = resources
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    result["execution"] = {"git_base_revision": revision, "source_sha256": {
        path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest() for path in
        (Path(__file__).resolve(), Path(archive.__file__), Path(native.__file__), Path(projection.__file__),
         ROOT / "tools/read_thin_device_grid_AP_pnc.py", ROOT / "src/phononic/workflow_AP_pnc.py")},
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform()}
    if args.output is None:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
