"""Check byte-level historical source provenance. Author: Athul Prem | _AP_pnc."""

from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_phase_refinement_AP_pnc as evidence


class HistoricalSourceTests(unittest.TestCase):
    def test_current_archive_and_corruption(self):
        relative, historical_sha = next(iter(evidence.ARCHIVED_SOURCE_HASHES.items()))
        archive = Path("provenance/solver_sources_AP_pnc") / f"{historical_sha}_AP_pnc.py"
        historical_bytes = (ROOT / archive).read_bytes()
        self.assertEqual(sha256(historical_bytes).hexdigest(), historical_sha)
        current_bytes = b"New scattering implementation.\n"
        with TemporaryDirectory() as directory, patch.object(evidence, "ROOT", Path(directory).resolve()):
            current = evidence.ROOT / relative
            current.parent.mkdir(parents=True)
            current.write_bytes(current_bytes)
            retained = evidence.ROOT / archive
            retained.parent.mkdir(parents=True)
            retained.write_bytes(historical_bytes)
            evidence.check_recorded_source(relative, sha256(current_bytes).hexdigest())
            evidence.check_recorded_source(relative, historical_sha)

            retained.write_bytes(historical_bytes + b"\n")
            with self.assertRaisesRegex(ValueError, "Evidence checksum differs"):
                evidence.check_recorded_source(relative, historical_sha)

    def test_unregistered_archive_is_rejected(self):
        relative = "src/phononic/scattering_AP_pnc.py"
        unrecognized = b"Unregistered version.\n"
        checksum = sha256(unrecognized).hexdigest()
        with TemporaryDirectory() as directory, patch.object(evidence, "ROOT", Path(directory).resolve()):
            archive = evidence.ROOT / "provenance/solver_sources_AP_pnc" / f"{checksum}_AP_pnc.py"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(unrecognized)
            with self.assertRaisesRegex(ValueError, "No retained numerical source matches"):
                evidence.check_recorded_source(relative, checksum)


if __name__ == "__main__":
    unittest.main()
