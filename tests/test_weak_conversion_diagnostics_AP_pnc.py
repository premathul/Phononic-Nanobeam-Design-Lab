"""Check saved weak-conversion diagnostics. Athul Prem | _AP_pnc."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from tools import weak_conversion_diagnostics_AP_pnc as diagnostics


ROOT = Path(__file__).resolve().parents[1]


class WeakConversionDiagnosticsTests(unittest.TestCase):
    def test_saved_reciprocal_powers_and_complex_cross_ratios(self):
        expected = (
            ("h100 z4", (1302, 1303, 1282, 1283), (1.6551626342101197e-6, 1.6551626358499875e-6),
             9.907593498049502e-10, -0.004120368770518241 - 0.0006581997306077585j,
             0.004172608978856529, -170.92408570055966),
            ("h80 z4", (1302, 1303, 1282, 1283), (9.916496528595719e-8, 9.91649649635775e-8),
             3.250943453296039e-9, -8.047642358871812e-05 + 0.00030186380101409395j,
             0.00031240712078361353, 104.92776845332521),
            ("h80 z6", (1877, 1878, 1851, 1852), (7.969107643009675e-7, 7.969107666696145e-7),
             2.972286441061089e-9, -0.0025730434115938663 + 0.000969841971540637j,
             0.002749753779469797, 159.34736670500845),
        )
        rows = diagnostics.load_diagnostics()
        for row, (mesh, roots, powers, difference, ratio, magnitude, angle) in zip(rows, expected, strict=True):
            with self.subTest(mesh=mesh):
                self.assertEqual(row["mesh"], mesh)
                self.assertEqual(row["frequency_hz"], 887_500_000)
                self.assertEqual((row["mode_1_incident_root"], row["mode_1_reflected_root"],
                                  row["mode_5_incident_root"], row["mode_5_reflected_root"]), roots)
                self.assertEqual((row["power_1_to_5"], row["power_5_to_1"]), powers)
                self.assertEqual(row["mean_power"], sum(powers) / 2)
                self.assertAlmostEqual(row["reciprocity_relative_difference"], difference, delta=1e-18)
                self.assertLessEqual(row["reciprocity_relative_difference"], 3.3e-9)
                self.assertAlmostEqual(row["cross_ratio_real"], ratio.real, delta=1e-15)
                self.assertAlmostEqual(row["cross_ratio_imag"], ratio.imag, delta=1e-15)
                self.assertAlmostEqual(row["cross_ratio_abs"], magnitude, delta=1e-15)
                self.assertAlmostEqual(row["cross_ratio_phase_deg"], angle, delta=1e-10)
                for name, power in zip(("r21", "r12"), powers, strict=True):
                    amplitude = complex(row[f"{name}_real"], row[f"{name}_imag"])
                    self.assertTrue(isclose(abs(amplitude) ** 2, power, rel_tol=1e-14))
                source = ROOT / row["portable_control_source"]
                self.assertEqual(sha256(source.read_bytes()).hexdigest(), row["portable_control_sha256"])

    def test_nonzero_complex_row_and_column_scaling_preserves_the_ratio(self):
        ratio = diagnostics.reflection_cross_ratio
        r11, r12, r21, r22 = (0.11 + .02j, -.001 - .002j, -.0003 - .0004j, -.004 + .002j)
        a1, a2 = 1.5 + 2j, -.37j
        b1, b2 = -1.1 - .23j, .06 + .5j
        native = ratio(r11, r12, r21, r22)
        scaled = ratio(a1 * r11 * b1, a1 * r12 * b2, a2 * r21 * b1, a2 * r22 * b2)
        self.assertAlmostEqual(abs(scaled - native), 0.0, delta=1e-17)
        for block in ((0j, r12, r21, r22), (r11, r12, r21, 0j)):
            with self.assertRaisesRegex(ValueError, "nonzero diagonal"):
                ratio(*block)

    def test_changed_saved_coefficient_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source in (diagnostics.MANIFEST, diagnostics.INPLANE_SOURCE, diagnostics.THICKNESS_SOURCE):
                target = root / source
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / source).read_bytes())

            # Isolated Python uses the compact files and standard library alone.
            script = (
                "import pathlib, runpy, sys; "
                "load = runpy.run_path(sys.argv[1])['load_diagnostics']; "
                "assert len(load(pathlib.Path(sys.argv[2]))) == 3"
            )
            completed = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script,
                                        diagnostics.__file__, str(root)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)

            target = root / diagnostics.INPLANE_SOURCE
            control = json.loads(target.read_bytes())
            first = control["raw_responses"]["reference_h100"]["coefficients"][0]
            first["amplitude"][0] *= 2
            raw = json.dumps(control).encode()
            target.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "differs from the dataset manifest"):
                diagnostics.load_diagnostics(root)

            manifest_path = root / diagnostics.MANIFEST
            manifest = json.loads(manifest_path.read_bytes())
            source = next(control["source"] for control in manifest["controls"]
                          if control["source"]["path"] == diagnostics.INPLANE_SOURCE)
            source["sha256"] = sha256(raw).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Inconsistent reflection amplitude and power"):
                diagnostics.load_diagnostics(root)


if __name__ == "__main__":
    unittest.main()
