"""Check the saved short-guide acceptance inputs without solving. Author: Athul Prem | _AP_pnc."""

import copy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_scattering_AP_pnc import (
    REFINEMENT_STEM, SAFE_RECONSTRUCTION_SOURCES, SCATTERING_SOURCE, read_acceptance_inputs, source_identity,
)


class SavedScatteringValidationTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "verification" / f"{REFINEMENT_STEM}.json"
        self.saved = json.loads(self.path.read_bytes())
        self.identity = source_identity()

    def test_archived_scattering_baseline_remains_usable(self):
        case, provenance = read_acceptance_inputs(self.path, self.identity, self.saved)
        self.assertEqual(case, self.saved["cases"]["transverse_refined"])
        self.assertEqual(provenance["original"], self.saved["baseline_evidence"])
        refinement = provenance["refinement"]
        self.assertEqual(set(refinement["matching_numerical_source_sha256"]),
                         {f"src/phononic/{name}" for name in SAFE_RECONSTRUCTION_SOURCES})
        self.assertEqual(refinement["scattering_source"], {
            "path": SCATTERING_SOURCE,
            "saved_sha256": self.saved["implementation"]["source_sha256"][SCATTERING_SOURCE],
            "current_candidate_sha256": self.identity["source_sha256"][SCATTERING_SOURCE],
        })

    def test_unknown_scattering_source_is_rejected(self):
        forged = copy.deepcopy(self.saved)
        forged["implementation"]["source_sha256"][SCATTERING_SOURCE] = "0" * 64
        with TemporaryDirectory() as directory:
            path = Path(directory) / self.path.name
            path.write_text(json.dumps(forged), encoding="utf-8")
            shutil.copyfile(self.path.with_suffix(".csv"), path.with_suffix(".csv"))
            with self.assertRaisesRegex(ValueError, "No retained numerical source matches"):
                read_acceptance_inputs(path, self.identity, self.saved)

    def test_safe_source_mismatch_is_rejected(self):
        self.identity["source_sha256"]["src/phononic/port_modes_AP_pnc.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Saved SAFE reconstruction source checksums differ"):
            read_acceptance_inputs(self.path, self.identity, self.saved)


if __name__ == "__main__":
    unittest.main()
