"""Exercise cross-host preparation and exact native resume. Author: Athul Prem | _AP_pnc."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from test_seed_frequency_map_AP_pnc import saved_bytes, small_guide
from tools import transfer_frequency_preparation_AP_pnc as transfer


class TransferPreparationTests(unittest.TestCase):
    def test_prepare_linux_map_then_resume_without_rebuilding(self):
        native, workflow = transfer.native, transfer.workflow
        implementation = workflow.source_identity()
        windows = {**implementation, "platform": "Windows-fixture", "python": "3.12.14"}
        linux = {**implementation, "platform": "Linux-fixture", "python": "3.12.9"}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            donor, output, receipt = root / "donor", root / "donor_linux", root / "receipt.json"
            config = small_guide()
            with patch.object(workflow, "source_identity", return_value=windows):
                original = native.map_scattering(config, [0.25], donor, max_new_points=0)
                checksum = transfer.export_preparation(donor, receipt)
                with self.assertRaises(FileExistsError):
                    transfer.export_preparation(donor, receipt)
            original_bytes = saved_bytes(donor)

            changed = deepcopy(linux)
            changed["packages"]["numpy"] = "different"
            with patch.object(workflow, "source_identity", return_value=changed):
                with self.assertRaisesRegex(ValueError, "package versions differ"):
                    transfer.import_preparation(donor, receipt, checksum, output, [0.25, 0.30])
            self.assertFalse(output.exists())

            with (
                patch.object(workflow, "source_identity", return_value=linux),
                patch.object(workflow, "build_mesh", side_effect=AssertionError("rebuilt the transferred mesh")),
                patch.object(workflow, "_assemble_system", side_effect=AssertionError("reassembled body operators")),
                patch.object(workflow, "assemble_port", side_effect=AssertionError("reassembled the port section")),
                patch.object(workflow, "solve_port_modes", wraps=workflow.solve_port_modes) as modes,
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                    transfer.import_preparation(donor, receipt, "0" * 64, output, [0.25, 0.30])

                mass = Path(original["shared"]["artifacts"]["mass"]["path"])
                (donor / mass).write_bytes(original_bytes[mass] + b"changed")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    transfer.import_preparation(donor, receipt, checksum, output, [0.25, 0.30])
                (donor / mass).write_bytes(original_bytes[mass])
                self.assertFalse(output.exists())

                prepared = transfer.import_preparation(donor, receipt, checksum, output, [0.25, 0.30])
                modes.assert_not_called()
                self.assertEqual(prepared["shared"], original["shared"])
                self.assertEqual([point["status"] for point in prepared["points"]], ["pending", "pending"])
                reuse = prepared["preparation_reuse"]
                self.assertEqual(reuse["method"], transfer.METHOD)
                self.assertEqual(reuse["receipt_sha256"], checksum)
                self.assertEqual(reuse["receipt"]["donor_checkpoint_sha256"], original["checkpoint_sha256"])
                self.assertEqual(reuse["receipt"]["donor_numerical_identity"]["platform"], windows["platform"])
                self.assertEqual(prepared["inputs"]["numerical_identity"]["platform"], linux["platform"])
                for artifact in original["shared"]["artifacts"].values():
                    path = Path(artifact["path"])
                    self.assertEqual((output / path).read_bytes(), original_bytes[path])
                    self.assertFalse((output / path).samefile(donor / path))

                resumed = native.map_scattering(config, [0.25, 0.30], output, max_new_points=1)
                self.assertEqual(resumed["preparation_reuse"], reuse)
                self.assertEqual([point["status"] for point in resumed["points"]], ["accepted", "pending"])
                self.assertEqual([call.args[1] for call in modes.call_args_list], [250e6])
            self.assertEqual(saved_bytes(donor), original_bytes)

            with patch.object(workflow, "source_identity", return_value=windows):
                with self.assertRaisesRegex(ValueError, "source or environment"):
                    native.map_scattering(config, [0.25, 0.30], output, max_new_points=1)


if __name__ == "__main__":
    unittest.main()
