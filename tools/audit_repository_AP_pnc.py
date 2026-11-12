#!/usr/bin/env python3
"""Read-only archive inventory. Author: Athul Prem | _AP_pnc."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.archive_runtime_AP_pnc import audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete-archive", action="store_true")
    parser.add_argument("--list-missing", action="store_true", help="Include every unresolved original path")
    args = parser.parse_args()
    try:
        result = audit()
        if not args.list_missing:
            result.pop("unresolved_entries")
        code = 1 if result["integrity_errors"] else 2 if args.require_complete_archive and not result["archive_complete"] else 0
        result["status"] = "FAIL_INTEGRITY" if code == 1 else "BLOCKED_INCOMPLETE_ARCHIVE" if code == 2 else "PASS_UPLOAD_INTEGRITY"
        print(json.dumps(result, indent=2))
        return code
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL_INTEGRITY", "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
