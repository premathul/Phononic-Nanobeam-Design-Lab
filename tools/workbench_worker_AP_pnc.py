"""Headless desktop workbench job runner. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def write_response(path: Path, response: dict) -> None:
    """Publish one complete response without replacing an existing job record."""
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".response-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(response, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            # Windows rename fails if the destination exists.
            temporary.rename(path)
        else:
            # POSIX rename replaces existing files. Linking publishes without overwriting.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--action", choices=("mesh", "run", "compare"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.response.exists() or args.response.is_symlink():
        print(f"The job response already exists: {args.response}", file=sys.stderr)
        return 1
    if not args.response.parent.is_dir():
        print(f"The job response directory does not exist: {args.response.parent}", file=sys.stderr)
        return 1

    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    response = {"action": args.action}
    try:
        if args.action == "compare":
            if args.reference is None or args.candidate is None or args.config is not None:
                raise ValueError("Comparison requires --reference and --candidate. Omit --config.")
            from compare_scattering_AP_pnc import compare_runs

            summary = compare_runs(args.reference, args.candidate, args.output_root)
        else:
            if args.config is None or args.reference is not None or args.candidate is not None:
                raise ValueError("Mesh checks and solves require --config. Omit --reference and --candidate.")
            config = json.loads(args.config.read_text(encoding="utf-8-sig"))
            if not isinstance(config, dict):
                raise ValueError("The configuration must be a JSON object.")
            from phononic.workflow_AP_pnc import preflight_mesh, run_study

            execute = preflight_mesh if args.action == "mesh" else run_study
            summary = execute(config, args.output_root)
        response.update(status=summary["status"], summary=summary)
        # A cold result can contain NumPy scalar values even though the
        # workflow's persisted summary has already normalized them to JSON.
        from numpy import generic

        def normalize_scalar(value):
            if isinstance(value, generic):
                return value.item()
            raise TypeError(f"Unsupported JSON value: {type(value).__name__}")

        response = json.loads(json.dumps(response, ensure_ascii=False, allow_nan=False, default=normalize_scalar))
    except (Exception, KeyboardInterrupt) as exc:
        error = "The job was interrupted." if isinstance(exc, KeyboardInterrupt) else str(exc) or type(exc).__name__
        response = {"action": args.action, "status": "failed", "error": error}

    try:
        write_response(args.response, response)
    except OSError as exc:
        print(f"Could not write the job response: {exc}", file=sys.stderr)
        return 1
    return 1 if response["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
