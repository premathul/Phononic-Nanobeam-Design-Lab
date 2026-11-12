"""Archive-path and provenance guard tests. Author: Athul Prem | _AP_pnc."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.archive_runtime_AP_pnc import (
    audit, checked_path, relative_path, require_new_run, restore, source_for, validate_run_name,
)


class ArchiveGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.raw = b"preserved original numerical evidence\n"
        (self.root / "evidence_AP_pnc.txt").write_bytes(self.raw)
        self.row = {
            "original": "legacy/input.txt", "flat": "legacy__input.txt",
            "repository_path": "evidence_AP_pnc.txt", "status": "available_exact",
            "sha256": hashlib.sha256(self.raw).hexdigest(),
        }

    def test_rejects_traversal_and_ambiguous_paths(self):
        for path in ("../escape", "/absolute", "a/../../b", "a\\b", "C:drive", "a//b", "./a", "a/./b", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative_path(path)

    def test_symlink_source_and_parent_are_rejected(self):
        (self.root / "alias").symlink_to(self.root / "evidence_AP_pnc.txt")
        with self.assertRaises(ValueError):
            checked_path(self.root, "alias")
        (self.root / "linked_directory").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            checked_path(self.root, "linked_directory/evidence_AP_pnc.txt")

    def test_corrupted_hash_blocks_restore_before_writes(self):
        (self.root / "evidence_AP_pnc.txt").write_bytes(b"changed")
        out = Path(self.temp.name) / "restored"
        with self.assertRaises(ValueError):
            restore([self.row], out, self.root)
        self.assertFalse(out.exists())

    def test_all_inputs_checked_before_partial_restore(self):
        missing = dict(self.row, original="legacy/missing.txt", repository_path=None, status="missing")
        out = Path(self.temp.name) / "restored"
        with self.assertRaises(FileNotFoundError):
            restore([self.row, missing], out, self.root)
        self.assertFalse(out.exists())

    def test_exact_hash_alias_restores_original_paths(self):
        alias = dict(self.row, original="another/input.txt", status="available_exact_hash_alias")
        out = Path(self.temp.name) / "restored"
        restore([self.row, alias], out, self.root)
        self.assertEqual((out / "legacy/input.txt").read_bytes(), self.raw)
        self.assertEqual((out / "another/input.txt").read_bytes(), self.raw)
        self.assertEqual((self.root / "evidence_AP_pnc.txt").read_bytes(), self.raw)

    def test_existing_restore_target_is_never_overwritten(self):
        out = Path(self.temp.name) / "restored"
        (out / "legacy").mkdir(parents=True)
        existing = out / "legacy/input.txt"
        existing.write_bytes(b"existing result")
        with self.assertRaises(ValueError):
            restore([self.row], out, self.root)
        self.assertEqual(existing.read_bytes(), b"existing result")

    def test_missing_archive_does_not_turn_integrity_into_completeness(self):
        missing = dict(self.row, original="legacy/missing.txt", repository_path=None, status="missing")
        provenance = self.root / "provenance"
        provenance.mkdir()
        (provenance / "archive_catalog_AP_pnc.json").write_text(json.dumps({"entries": [self.row, missing]}))
        (provenance / "relocation_manifest_AP_pnc.json").write_text(json.dumps({"files": [self.row]}))
        report = audit(self.root)
        self.assertTrue(report["uploaded_bytes_preserved"])
        self.assertFalse(report["archive_complete"])
        self.assertEqual(report["unresolved_expected_entries"], 1)

    def test_run_name_and_existing_output_guards(self):
        for name in ("../other", "/absolute", "a b", "a" * 41, "-starts-with-dash"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_run_name(name)
        target = require_new_run("example_01", self.root)
        self.assertFalse(target.exists())
        target.mkdir(parents=True)
        with self.assertRaises(ValueError):
            require_new_run("example_01", self.root)


if __name__ == "__main__":
    unittest.main()
