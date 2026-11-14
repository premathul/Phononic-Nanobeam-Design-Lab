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
RESUMABLE_ACTIONS = ("run-resumable", "resume")


def write_response(path: Path, response: dict, *, replace: bool = False) -> None:
    """Publish progress or a write-once job response atomically."""
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".response-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(response, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            temporary.replace(path)
        elif os.name == "nt":
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
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--action", choices=("mesh", "run", "compare", *RESUMABLE_ACTIONS), required=True)
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
        resumable = args.action in RESUMABLE_ACTIONS
        if not resumable and (args.resume_from is not None or args.wall_seconds is not None):
            raise ValueError("--resume-from and --wall-seconds apply to resumable scattering.")
        if args.action == "compare":
            if args.reference is None or args.candidate is None or args.config is not None:
                raise ValueError("Comparison requires --reference and --candidate. Omit --config.")
            from compare_scattering_AP_pnc import compare_runs

            summary = compare_runs(args.reference, args.candidate, args.output_root)
        else:
            if args.reference is not None or args.candidate is not None:
                raise ValueError("--reference and --candidate apply to saved-run comparisons.")
            if args.action == "resume":
                if args.resume_from is None or args.config is not None:
                    raise ValueError("Resume requires --resume-from. Omit --config.")
                config = None
            else:
                if args.config is None or args.resume_from is not None:
                    raise ValueError("Mesh checks and new solves require --config. Omit --resume-from.")
                config = json.loads(args.config.read_text(encoding="utf-8-sig"))
                if not isinstance(config, dict):
                    raise ValueError("The configuration must be a JSON object.")

            if resumable:
                from phononic.resumable_scattering_AP_pnc import run_resumable_scattering

                progress_path = args.response.with_suffix(".progress.json")
                cancel_path = args.response.with_suffix(".cancel")

                def publish_progress(summary: dict) -> None:
                    try:
                        write_response(progress_path, {"action": args.action, "status": summary["status"],
                                                      "summary": summary}, replace=True)
                    except OSError as exc:
                        print(f"Could not publish worker progress: {exc}", file=sys.stderr)

                arguments = {"resume_from": args.resume_from} if config is None else {
                    "config": config, "output_root": args.output_root,
                }
                if args.wall_seconds is not None:
                    arguments["wall_seconds"] = args.wall_seconds
                summary = run_resumable_scattering(**arguments, progress_callback=publish_progress,
                                                    should_cancel=cancel_path.exists)
            else:
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
    return 1 if response["status"] in ("blocked", "failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
