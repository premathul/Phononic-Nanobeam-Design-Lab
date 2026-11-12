#!/usr/bin/env python3
"""Verify available archive evidence without solving FEM. Author: Athul Prem | _AP_pnc."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unittest

sys.dont_write_bytecode = True
from tools.archive_runtime_AP_pnc import ROOT, audit, digest, load_catalog, load_json, source_for

SMOKE = "results/4K/run_new_verified_smoke/"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_available_smoke(*, fields: bool = False) -> dict:
    catalog = load_catalog()

    def original(relative: str):
        return source_for(SMOKE + relative, catalog)

    stage = load_json(original("logs/stage_smoke.json"))
    require(all(stage.get(k) == "PASS" for k in ("four_GHz", "five_frequencies", "operator_regression")),
            "Saved smoke success markers differ")
    outputs = stage["_output_sha256"]
    require(len(outputs) == 26, "Expected the 26 outputs of the uploaded verified-smoke stage")
    for relative, expected in outputs.items():
        require(digest(original(relative)) == expected, "Saved stage output differs: " + relative)
    with original("tables/five_frequency_comparison.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 5, "Expected five saved frequencies")
    max_residual = max_sponge_balance = max_ratio_error = 0.0
    for row, frequency in zip(rows, (2, 4, 6, 8, 10)):
        require(float(row["frequency_GHz"]) == frequency, "Saved frequency grid differs")
        for structure in ("reference", "periodic", "defect"):
            residual = float(row[structure + "_residual"])
            balance = float(row[structure + "_power_balance_relative_error"])
            probes = [float(row[structure + "_" + key]) for key in ("P1_u_m", "P3_u_m")]
            require(all(math.isfinite(x) for x in [residual, balance, *probes]), "Nonfinite saved value")
            require(0 <= residual < 1e-7, "Saved algebraic residual failed")
            require(0 <= balance < 1e-6, "Saved source/sponge balance failed")
            require(min(probes) > 0, "Saved probe amplitude must be positive")
            max_residual = max(max_residual, residual)
            max_sponge_balance = max(max_sponge_balance, balance)
            if structure != "reference":
                expected = 20 * math.log10(float(row[structure + "_P3_u_m"]) / float(row["reference_P3_u_m"]))
                actual = float(row[structure + "_relative_dB"])
                require(math.isfinite(actual), "Nonfinite saved displacement ratio")
                error = abs(actual - expected)
                require(error < 1e-10, "Saved displacement-ratio arithmetic differs")
                max_ratio_error = max(max_ratio_error, error)
    operator = load_json(original("tables/independent_operator_checks.json"))
    require(bool(operator), "Empty saved operator comparison")
    require(all(math.isfinite(float(x)) and 0 <= float(x) < 1e-8 for x in operator.values()),
            "Saved operator comparison failed")
    result = {
        "status": "PASS_AVAILABLE_SAVED_SMOKE",
        "source_run": SMOKE.rstrip("/"),
        "saved_stage_outputs_hashed": len(outputs),
        "saved_frequencies_checked": len(rows),
        "saved_structure_frequency_records_checked": 3 * len(rows),
        "maximum_saved_algebraic_residual": max_residual,
        "maximum_saved_source_sponge_balance_error": max_sponge_balance,
        "maximum_displacement_ratio_arithmetic_error_dB": max_ratio_error,
        "saved_operator_comparisons_checked": len(operator),
        "calibrated_power_transmission_validated": False,
        "full_333_solve_60_mode_archive_reverified": False,
        "simulation_rerun": False,
    }
    if fields:
        import numpy as np
        with np.load(original("fields/field_4GHz.npz"), allow_pickle=False) as saved:
            require(all(np.isfinite(saved[key]).all() for key in saved.files), "Nonfinite saved field array")
            points = saved["points_m"]
            triangles = saved["triangles"]
            displacement = saved["displacement_complex"]
            strain = saved["strain_energy_density"]
            require(points.ndim == 2 and points.shape[1] == 2, "Expected in-plane point coordinates")
            require(triangles.ndim == 2 and triangles.shape[1] == 3 and triangles.size > 0, "Invalid triangle shape")
            require(np.issubdtype(triangles.dtype, np.integer), "Triangle connectivity must be integer")
            require(triangles.min() >= 0 and triangles.max() < len(points), "Triangle index outside point array")
            require(displacement.shape == (2 * len(points),), "Displacement shape differs")
            require(strain.shape == (len(triangles),), "Strain-energy shape differs")
            result["saved_field_check"] = {
                "status": "PASS_FINITE_ARRAYS_AND_TOPOLOGY", "points": len(points),
                "triangles": len(triangles), "arrays_checked": len(saved.files),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", action="store_true", help="Check the available 4 GHz saved arrays; requires NumPy")
    parser.add_argument("--safety", action="store_true", help="Run path/hash/archive-guard unit tests")
    parser.add_argument("--require-complete-archive", action="store_true")
    parser.add_argument("--compare-name", help="Legacy compatibility option; blocked until the original full baseline is recovered")
    args = parser.parse_args()
    try:
        report = audit()
        require(not report["integrity_errors"], "Uploaded archive integrity failed: " + json.dumps(report["integrity_errors"]))
        report.pop("unresolved_entries")
        report["available_saved_smoke"] = verify_available_smoke(fields=args.fields)
        if args.safety:
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_archive_runtime_AP_pnc.py")
            tested = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
            require(tested.testsRun > 0 and tested.wasSuccessful(), "Archive guard tests failed or no tests found")
            report["archive_guard_tests"] = {"status": "PASS", "count": tested.testsRun}
        code = 0
        report["status"] = "PASS_AVAILABLE_EVIDENCE"
        if args.compare_name:
            code = 2
            report["status"] = "BLOCKED_FRESH_COMPARISON"
            report["comparison_error"] = "Full latest baseline and executable reproduction workflow require recovery; no fresh-run comparison was performed."
        elif args.require_complete_archive and not report["archive_complete"]:
            code = 2
            report["status"] = "BLOCKED_INCOMPLETE_ARCHIVE"
        print(json.dumps(report, indent=2))
        return code
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(json.dumps({"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "FAIL_VERIFICATION", "error": str(exc), "simulation_rerun": False}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
