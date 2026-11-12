#!/usr/bin/env python3
"""Inspect the legacy FEM run prerequisites. Author: Athul Prem | _AP_pnc.

The GitHub upload is incomplete. This entrypoint deliberately performs no FEM
execution: it gives an honest dry-run plan or a nonzero blocked status. Recover
the original inputs, validate their hashes, and implement/review an execution
adapter before enabling reproduction. The new 3D campaign is planned work.
"""
from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True
from tools.archive_runtime_AP_pnc import audit, require_new_run, run_path, validate_run_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--stage", choices=("smoke", "sweep", "bands", "cavity", "robustness", "all"), default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--excel-python", help="Recorded for the eventual recovered legacy runner; no process is launched")
    parser.add_argument("--postprocess", choices=("verify_eigen_scaling", "check_legacy_bloch", "finalize"))
    args = parser.parse_args()
    try:
        validate_run_name(args.name)
        if args.postprocess and not args.resume:
            raise ValueError("--postprocess requires --resume")
        output = run_path(args.name) if args.resume else require_new_run(args.name)
        if args.resume and not output.is_dir():
            raise ValueError("No collected run exists to resume")
        report = audit()
        if report["integrity_errors"]:
            print(json.dumps({"status": "FAIL_INTEGRITY", "errors": report["integrity_errors"], "simulation_rerun": False}, indent=2))
            return 1
        result = {
            "author": "Athul Prem", "file_tag": "_AP_pnc",
            "status": "DRY_RUN_BLOCKED" if args.dry_run else "BLOCKED_NO_EXECUTION",
            "run_name": args.name, "requested_stage": args.stage,
            "planned_output": "results/runs/" + args.name,
            "archive_complete": report["archive_complete"],
            "unresolved_expected_entries": report["unresolved_expected_entries"],
            "critical_missing_runner_inputs": report["critical_missing_runner_inputs"],
            "execution_adapter_enabled": False,
            "next_action": "Recover exact archived source/configuration, rerun strict verification, then review an execution adapter before starting the legacy smoke case.",
            "new_3D_campaign_implemented": False,
            "simulation_rerun": False, "files_created": False,
        }
        print(json.dumps(result, indent=2))
        return 0 if args.dry_run else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED_NO_EXECUTION", "error": str(exc), "simulation_rerun": False, "files_created": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
