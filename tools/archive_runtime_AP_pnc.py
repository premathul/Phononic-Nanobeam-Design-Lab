"""Preserve and resolve the uploaded archive. Author: Athul Prem | _AP_pnc.

This module contains file/provenance operations, not an elastic-wave solver.
Historical source is restored only after all expected bytes are available.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = "Athul Prem"
FILE_TAG = "_AP_pnc"
CATALOG = "provenance/archive_catalog_AP_pnc.json"
RELOCATIONS = "provenance/relocation_manifest_AP_pnc.json"
CRITICAL_RUN_INPUTS = (
    "handoff.py",
    "config.py",
    "data/nanobeam_hole_array.xlsx",
    "data/hole_geometry_derived.csv",
    "scripts/solve_direct_fem.py",
    *("results/4K/portable_template/" + name for name in (
        "run.py", "numerics.py", "config.json", "MATERIAL_PROVENANCE.md",
        "NUMERICAL_METHODS.md", "finalize.py", "verify_eigen_scaling.py",
        "check_legacy_bloch.py",
    )),
)


def relative_path(value: str) -> Path:
    """Reject ambiguous and escaping paths before any filesystem operation."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or ":" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*pure.parts)


def checked_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    """Resolve within a real directory; reject all symlink components."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("A symlink cannot be an archive root")
    base = root.resolve()
    path = base
    for part in relative_path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink not permitted: {relative}")
    if not path.resolve().is_relative_to(base):
        raise ValueError(f"Path escapes archive root: {relative}")
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Missing file: {relative}")
    return path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_catalog(root: Path = ROOT) -> list[dict]:
    rows = load_json(checked_path(root, CATALOG))["entries"]
    seen = set()
    for row in rows:
        relative_path(row["original"])
        if row["original"] in seen:
            raise ValueError("Duplicate original path: " + row["original"])
        seen.add(row["original"])
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid archive SHA-256")
        if row.get("repository_path") is not None:
            relative_path(row["repository_path"])
    return rows


def source_for(original: str, rows: list[dict] | None = None, root: Path = ROOT) -> Path:
    relative_path(original)
    rows = load_catalog(root) if rows is None else rows
    matches = [row for row in rows if row["original"] == original]
    if len(matches) != 1 or not matches[0].get("repository_path"):
        raise FileNotFoundError("Original archive input unavailable: " + original)
    row = matches[0]
    path = checked_path(root, row["repository_path"])
    if digest(path) != row["sha256"]:
        raise ValueError("Archive hash differs: " + row["repository_path"])
    return path


def audit(root: Path = ROOT) -> dict:
    relocations = load_json(checked_path(root, RELOCATIONS))["files"]
    errors = []
    seen = set()
    for row in relocations:
        try:
            path = checked_path(root, row["repository_path"])
            if row["repository_path"] in seen:
                raise ValueError("Duplicate relocation path")
            seen.add(row["repository_path"])
            if digest(path) != row["sha256"]:
                raise ValueError("SHA-256 differs from uploaded baseline")
        except (OSError, ValueError) as exc:
            errors.append({"repository_path": row["repository_path"], "error": str(exc)})
    rows = load_catalog(root)
    unavailable = []
    aliases = available = 0
    for row in rows:
        if row.get("repository_path") is None:
            unavailable.append({"original": row["original"], "flat": row["flat"], "status": row["status"]})
            continue
        try:
            source_for(row["original"], rows, root)
            available += 1
            aliases += row["status"] == "available_exact_hash_alias"
        except (OSError, ValueError) as exc:
            errors.append({"original": row["original"], "error": str(exc)})
    by_original = {row["original"]: row for row in rows}
    critical_missing = [name for name in CRITICAL_RUN_INPUTS
                        if not by_original.get(name, {}).get("repository_path")]
    return {
        "author": AUTHOR,
        "file_tag": FILE_TAG,
        "uploaded_files_checked": len(relocations),
        "uploaded_bytes_preserved": not errors,
        "archive_manifest_entries": len(rows),
        "available_expected_entries": available,
        "exact_hash_aliases": aliases,
        "unresolved_expected_entries": len(unavailable),
        "archive_complete": not unavailable and not errors,
        "critical_missing_runner_inputs": critical_missing,
        "unresolved_entries": unavailable,
        "integrity_errors": errors,
        "simulation_rerun": False,
    }


def restore(rows: list[dict], destination: Path, root: Path = ROOT) -> None:
    """Validate every input first, then reconstruct original paths in a new tree."""
    destination = Path(destination)
    prepared = []
    targets = set()
    for row in rows:
        src = source_for(row["original"], rows, root)
        target = checked_path(destination, row["original"], must_exist=False)
        if target in targets or target.exists():
            raise ValueError("Refusing to overwrite restored path: " + row["original"])
        targets.add(target)
        prepared.append((src, target))
    for src, target in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


@contextmanager
def project(root: Path = ROOT):
    """Temporary full original layout; never fill missing inputs with guesses."""
    report = audit(root)
    if not report["archive_complete"]:
        raise ValueError(f"Archive incomplete: {report['unresolved_expected_entries']} unresolved expected entries")
    with tempfile.TemporaryDirectory(prefix="athul_pnc_reproduction_") as name:
        work = Path(name) / "project"
        work.mkdir()
        restore(load_catalog(root), work, root)
        yield work


def validate_run_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", name):
        raise ValueError("Use a run label of 1-40 letters, digits, underscores or hyphens; begin with a letter or digit")
    return name


def run_path(name: str, root: Path = ROOT) -> Path:
    return checked_path(root, "results/runs/" + validate_run_name(name), must_exist=False)


def require_new_run(name: str, root: Path = ROOT) -> Path:
    path = run_path(name, root)
    if path.exists():
        raise ValueError("Run name already exists; choose a unique label")
    return path
