"""Exercise preparation reuse on a small native map. Author: Athul Prem | _AP_pnc."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import map_scattering_AP_pnc as mapper
from tools import seed_frequency_map_AP_pnc as seeder
from test_frequency_map_AP_pnc import independent_scattering, LENGTH_M, small_guide


def saved_bytes(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


class SeedFrequencyMapTests(unittest.TestCase):
    def test_cli_reuses_preparation_and_completes_with_native_resume(self):
        with TemporaryDirectory() as temporary:
            donor = Path(temporary) / "donor"
            output = Path(temporary) / "output"
            original = mapper.map_scattering(small_guide(), [0.25], donor)
            self.assertEqual(original["status"], "complete")
            donor_bytes = saved_bytes(donor)

            workflow = mapper.workflow
            with (
                patch.object(workflow, "build_mesh", side_effect=AssertionError("rebuilt the donor mesh")),
                patch.object(workflow, "_assemble_system", side_effect=AssertionError("reassembled body operators")),
                patch.object(workflow, "assemble_port", side_effect=AssertionError("reassembled the port section")),
                patch.object(workflow, "solve_port_modes", wraps=workflow.solve_port_modes) as modes,
                patch.object(workflow, "solve_scattering", wraps=workflow.solve_scattering) as solve,
                patch("builtins.print"),
            ):
                args = ["--from-map", str(donor), "--output", str(output), "--frequencies-ghz", "0.30", "0.35"]
                self.assertEqual(seeder.main(args), 0)
                partial = mapper.read_saved_map(output)
                self.assertEqual([point["status"] for point in partial["points"]], ["accepted", "pending"])
                self.assertEqual(partial["points"][1]["attempts"], [])
                self.assertEqual(partial["shared"], original["shared"])
                for artifact in partial["shared"]["artifacts"].values():
                    path = Path(artifact["path"])
                    self.assertEqual((output / path).read_bytes(), donor_bytes[path])
                    self.assertFalse((output / path).samefile(donor / path))

                checkpoint = mapper.read_map_checkpoint(output)
                reuse = checkpoint["preparation_reuse"]
                self.assertEqual(reuse["donor_checkpoint_sha256"], sha256(donor_bytes[Path(mapper.CHECKPOINT)]).hexdigest())
                self.assertEqual(reuse["donor_shared_sha256"], workflow.digest(original["shared"]))
                self.assertEqual(reuse["donor_source_git_commit"], original["source_git_commit"])
                self.assertEqual(reuse["seeder_source_sha256"], sha256(Path(seeder.__file__).read_bytes()).hexdigest())
                self.assertEqual(partial["preparation_reuse"], reuse)
                self.assertEqual(checkpoint["inputs"]["numerical_identity"], original["inputs"]["numerical_identity"])

                accepted = partial["points"][0]["accepted"]["summary"]
                first_summary = output / accepted["path"]
                first_bytes = first_summary.read_bytes()
                inputs = checkpoint["inputs"]
                self.assertEqual([call.args[1] for call in modes.call_args_list], [300e6])
                self.assertEqual([call.args[2] for call in solve.call_args_list], [300e6])
                completed = mapper.map_scattering(inputs["configuration"], inputs["frequencies_GHz"], output,
                                                   max_new_points=1)
                self.assertEqual(completed["status"], "complete")
                self.assertEqual(completed["preparation_reuse"], reuse)
                self.assertEqual([p["attempts"][0]["status"] for p in completed["points"]], ["accepted", "accepted"])
                self.assertEqual(completed["points"][0]["accepted"]["summary"], accepted)
                self.assertEqual(first_summary.read_bytes(), first_bytes)
                completed_bytes = saved_bytes(output)
                self.assertEqual(mapper.map_scattering(inputs["configuration"], inputs["frequencies_GHz"], output),
                                 completed)
                self.assertEqual(saved_bytes(output), completed_bytes)
                self.assertEqual([call.args[1] for call in modes.call_args_list], [300e6, 350e6])
                self.assertEqual([call.args[2] for call in solve.call_args_list], [300e6, 350e6])

            self.assertEqual(saved_bytes(donor), donor_bytes)
            summary = json.loads(first_bytes)
            self.assertTrue(summary["power_balance_passed"])
            self.assertEqual(summary["checkpointed_scattering"]["reference_planes_x_m"], [0., LENGTH_M])
            _, fresh_modes, reference, _ = independent_scattering(300e6)
            channels = completed["points"][0]["incoming_channels"]
            roots = reference.incident_mode_indices
            np.testing.assert_array_equal([row["port_index"] for row in channels], roots)
            current_wavenumbers = [row["k_rad_m"] for row in channels]
            np.testing.assert_allclose(current_wavenumbers, fresh_modes.wavenumbers_rad_m[roots].real, rtol=1e-9)
            donor_wavenumbers = [row["k_rad_m"] for row in original["points"][0]["incoming_channels"]]
            self.assertFalse(np.allclose(np.sort(current_wavenumbers), np.sort(donor_wavenumbers), rtol=1e-3))
            for quantity in ("reflection", "transmission"):
                np.testing.assert_allclose([row[quantity] for row in channels], getattr(reference, quantity),
                                           atol=1e-9, rtol=0)

    def test_prepared_donor_requires_the_published_checkpoint_digest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            donor = root / "donor"
            prepared = mapper.map_scattering(small_guide(), [0.25], donor, max_new_points=0)
            self.assertEqual(prepared["status"], "partial")
            self.assertEqual(prepared["points"][0]["status"], "pending")
            self.assertEqual(prepared["points"][0]["attempts"], [])
            donor_bytes = saved_bytes(donor)

            seeded = seeder.seed_frequency_map(donor, root / "output", [0.30])
            self.assertEqual(seeded["status"], "complete")
            self.assertEqual(seeded["shared"], prepared["shared"])
            self.assertEqual(seeded["preparation_reuse"]["donor_checkpoint_sha256"], prepared["checkpoint_sha256"])
            self.assertEqual(saved_bytes(donor), donor_bytes)

            # The checkpoint still passes its own content hash, but the publication pins its previous bytes.
            modified = mapper.read_map_checkpoint(donor)
            modified.pop("checkpoint_data_sha256")
            modified["source_git_commit"] = "0" * 40
            mapper._write_checkpoint(donor / mapper.CHECKPOINT, modified)
            self.assertEqual(mapper.read_map_checkpoint(donor)["source_git_commit"], "0" * 40)
            self.assertEqual(mapper.read_saved_map(donor), prepared)
            rejected = root / "rejected"
            with self.assertRaisesRegex(ValueError, "Published donor preparation differs from its checkpoint"):
                seeder.seed_frequency_map(donor, rejected, [0.30])
            self.assertFalse(rejected.exists())

    def test_refuses_changed_identity_damaged_operators_and_overlapping_destination(self):
        with TemporaryDirectory() as temporary:
            donor = Path(temporary) / "donor"
            output = Path(temporary) / "output"
            original = mapper.map_scattering(small_guide(), [0.25], donor)
            self.assertEqual(original["status"], "complete")
            donor_bytes = saved_bytes(donor)

            workflow = mapper.workflow
            current_source = workflow.source_identity()
            current_source["source_sha256"]["src/phononic/material_AP_pnc.py"] = "0" * 64
            mass = Path(original["shared"]["artifacts"]["mass"]["path"])
            with (
                patch.object(workflow, "build_mesh", side_effect=AssertionError("rebuilt a rejected mesh")),
                patch.object(workflow, "_assemble_system", side_effect=AssertionError("reassembled rejected operators")),
                patch.object(workflow, "assemble_port", side_effect=AssertionError("reassembled a rejected section")),
                patch.object(workflow, "solve_port_modes", side_effect=AssertionError("solved a rejected point")),
            ):
                with (
                    patch.object(workflow, "source_identity", return_value=current_source),
                    self.assertRaisesRegex(ValueError, "(?i)identity|source|environment"),
                ):
                    seeder.seed_frequency_map(donor, output, [0.30])
                self.assertFalse((output / mapper.CHECKPOINT).exists())

                (donor / mass).write_bytes(donor_bytes[mass] + b"changed")
                try:
                    with self.assertRaisesRegex(ValueError, "(?i)artifact|changed|checksum"):
                        seeder.seed_frequency_map(donor, output, [0.30])
                    self.assertFalse((output / mapper.CHECKPOINT).exists())
                    self.assertEqual((donor / mass).read_bytes(), donor_bytes[mass] + b"changed")
                finally:
                    (donor / mass).write_bytes(donor_bytes[mass])

                for overlapping in (donor, donor / "child", donor.parent):
                    with self.subTest(output=overlapping), self.assertRaises(ValueError):
                        seeder.seed_frequency_map(donor, overlapping, [0.30])
            self.assertEqual(saved_bytes(donor), donor_bytes)


if __name__ == "__main__":
    unittest.main()
