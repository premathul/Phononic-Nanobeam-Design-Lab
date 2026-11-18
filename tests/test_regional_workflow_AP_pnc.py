"""Public regional-mesh configuration and small workflow checks. Author: Athul Prem | _AP_pnc."""

import contextlib
import copy
from importlib.util import find_spec
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phononic.workflow_AP_pnc import main, preflight_mesh, resolve_config, run_study


def tiny_config():
    config = json.loads((ROOT / "configs/straight_scattering_AP_pnc.json").read_text())
    config["name"] = "regional_workflow_test_AP_pnc"
    config["geometry"].update(length_nm=1200, width_nm=400, thickness_nm=200)
    config["mesh"] = {"kind": "layered", "element_order": 1, "mesh_size_nm": 200,
                      "port_size_nm": 200, "thickness_layers": 1, "circle_resolution": 8}
    config["analysis"]["frequency_GHz"] = 0.1
    config["resources"].update(max_dofs=3000, max_port_dofs=100)
    return config


def regional_config():
    config = tiny_config()
    config["mesh"].update(lead_mesh_size_nm=150, patterned_x_interval_nm=[400, 800])
    return config


class RegionalConfiguration(unittest.TestCase):
    def test_omitted_controls_preserve_existing_mesh_configuration(self):
        for filename in ("straight_beam_AP_pnc.json", "perforated_beam_AP_pnc.json",
                         "perforated_layered_reference_AP_pnc.json"):
            with self.subTest(filename=filename):
                config = json.loads((ROOT / "configs" / filename).read_text())
                original = copy.deepcopy(config)
                resolved = resolve_config(config)
                self.assertEqual(config, original)
                self.assertEqual(resolved["mesh"], original["mesh"])
                self.assertNotIn("lead_mesh_size_nm", resolved["mesh"])
                self.assertNotIn("patterned_x_interval_nm", resolved["mesh"])

    def test_regional_controls_reject_invalid_or_incomplete_domains(self):
        config = regional_config()
        self.assertEqual(resolve_config(config)["mesh"], config["mesh"])
        for field in ("lead_mesh_size_nm", "patterned_x_interval_nm"):
            invalid = copy.deepcopy(config)
            del invalid["mesh"][field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, "together"):
                resolve_config(invalid)
        for value in (0, -1, True, float("nan"), float("inf"), "150", None):
            invalid = copy.deepcopy(config)
            invalid["mesh"]["lead_mesh_size_nm"] = value
            with self.subTest(lead_size=value), self.assertRaisesRegex(ValueError, "positive and finite"):
                resolve_config(invalid)
        for interval in ([0, 800], [400, 1200], [800, 400], [400, 400], [400],
                         [400, 800, 900], [True, 800], [400, float("nan")],
                         [400, float("inf")], "400,800", None):
            invalid = copy.deepcopy(config)
            invalid["mesh"]["patterned_x_interval_nm"] = interval
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "0 < start < end < length_nm"):
                resolve_config(invalid)
        for kind in ("gmsh", "structured"):
            invalid = copy.deepcopy(config)
            invalid["mesh"]["kind"] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "require mesh.kind=layered"):
                resolve_config(invalid)


@unittest.skipUnless(find_spec("gmsh"), "requires gmsh")
class RegionalWorkflow(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def test_cli_preflight_and_exact_cache_identity(self):
        config = tiny_config()
        source = self.output / "regional_config_AP_pnc.json"
        source.write_text(json.dumps(config), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--config", str(source), "--mesh-only", "--output-root", str(self.output),
                                   "--lead-mesh-size-nm", "150", "--patterned-x-interval-nm", "400", "800"]), 0)
        directory = next((self.output / "_meshes").iterdir())
        record = json.loads((directory / "summary_AP_pnc.json").read_text())
        self.assertEqual(record["mesh_inputs"]["settings"], regional_config()["mesh"])
        with np.load(directory / "mesh_AP_pnc.npz", allow_pickle=False) as arrays:
            xs = arrays["points_m"][0]
            for value in (0, 400e-9, 800e-9, 1200e-9):
                self.assertTrue(np.any(np.isclose(xs, value, rtol=0, atol=1e-20)))
        saved = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()}
        config["mesh"] = copy.deepcopy(record["mesh_inputs"]["settings"])
        with patch("phononic.workflow_AP_pnc.build_mesh", side_effect=AssertionError("exact mesh rebuilt")):
            reused = preflight_mesh(config, self.output)
        self.assertTrue(reused["reused"])
        self.assertEqual(reused["mesh_hash"], record["mesh_hash"])
        self.assertEqual(saved, {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()})
        hashes = {record["mesh_hash"]}
        for change in ({"lead_mesh_size_nm": 120}, {"patterned_x_interval_nm": [350, 850]}):
            different = copy.deepcopy(config)
            different["mesh"].update(change)
            new = preflight_mesh(different, self.output)
            self.assertFalse(new["reused"])
            self.assertNotIn(new["mesh_hash"], hashes)
            hashes.add(new["mesh_hash"])
        ordinary = preflight_mesh(tiny_config(), self.output)
        self.assertNotIn(ordinary["mesh_hash"], hashes)
        self.assertNotIn("lead_mesh_size_nm", ordinary["mesh_inputs"]["settings"])

    def test_small_regional_scattering_uses_checked_mesh_and_original_gates(self):
        config = regional_config()
        ready = preflight_mesh(config, self.output)
        self.assertLessEqual(ready["ndof"], 3000)
        self.assertLessEqual(max(ready["end_face_dofs"].values()), 100)
        with patch("phononic.workflow_AP_pnc.build_mesh", side_effect=AssertionError("checked mesh rebuilt")):
            result = run_study(config, self.output)
        self.assertEqual(result["mesh_checkpoint"]["mesh_hash"], ready["mesh_hash"])
        self.assertLess(result["maximum_relative_solver_residual"], 1e-8)
        self.assertLess(result["maximum_relative_port_residual"], 1e-8)
        self.assertTrue(result["power_balance_passed"])
        self.assertEqual(result["status"], "completed_unverified")
        self.assertFalse(result["matched_reference_passed"],
                         f"Coarse P1 matched-guide failure must remain explicit; max R={result['maximum_reflection']:.17g}")
        self.assertGreater(result["num_incident_channels"], 0)
        self.assertFalse(result["spatial_convergence_established"])
        print("Regional coarse P1 pilot (unverified): " + json.dumps({key: result[key] for key in (
            "ndof", "port_dofs", "num_incident_channels", "matched_reference_passed", "maximum_reflection",
            "maximum_relative_solver_residual", "maximum_relative_port_residual", "max_power_imbalance",
            "maximum_flux_power_imbalance", "maximum_modal_flux_difference", "maximum_net_outward_power_ratio")},
            default=lambda value: value.item()))
        with patch("phononic.workflow_AP_pnc.build_mesh", side_effect=AssertionError("solved mesh rebuilt")):
            self.assertTrue(run_study(config, self.output)["reused"])

    def test_small_quadratic_regional_scattering_matches_uniform_reference(self):
        config = regional_config()
        config["mesh"]["element_order"] = 2
        ready = preflight_mesh(config, self.output)
        self.assertLessEqual(ready["ndof"], 3000)
        self.assertLessEqual(max(ready["end_face_dofs"].values()), 100)
        with patch("phononic.workflow_AP_pnc.build_mesh", side_effect=AssertionError("checked P2 mesh rebuilt")):
            result = run_study(config, self.output)
        print("Regional P2 finite uniform benchmark (unverified): " + json.dumps({key: result[key] for key in (
            "ndof", "port_dofs", "num_incident_channels", "matched_reference_passed", "maximum_reflection",
            "maximum_relative_solver_residual", "maximum_relative_port_residual", "max_power_imbalance",
            "maximum_flux_power_imbalance", "maximum_modal_flux_difference", "maximum_net_outward_power_ratio")},
            default=lambda value: value.item()))
        self.assertEqual(result["mesh_checkpoint"]["mesh_hash"], ready["mesh_hash"])
        self.assertLess(result["maximum_relative_solver_residual"], 1e-8)
        self.assertLess(result["maximum_relative_port_residual"], 1e-8)
        self.assertTrue(result["power_balance_passed"])
        self.assertTrue(result["matched_reference_passed"],
                        f"Finite P2 uniform-guide criterion failed; max R={result['maximum_reflection']:.17g}")
        self.assertEqual(result["status"], "completed_unverified")
        self.assertGreater(result["num_incident_channels"], 0)
        self.assertFalse(result["spatial_convergence_established"])


if __name__ == "__main__":
    unittest.main()
