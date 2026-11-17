"""Plot independently checked saved section profiles. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix.lower() != ".png":
        raise ValueError("Select a fresh PNG output path.")
    diagnosis = json.loads(args.diagnosis.read_bytes())
    verification = json.loads(args.verification.read_bytes())
    if (verification["status"] != "independently_verified_saved_mirror_content"
            or verification["diagnosis"]["sha256"] != sha256(args.diagnosis)
            or diagnosis["frequency_hz"] != 1e9):
        raise ValueError("The plot requires the exact independently verified 1 GHz diagnosis.")
    section = diagnosis["sections"]["z6"]
    profiles = section["forward_profiles"]
    if (profiles["units"] != "m/sqrt(W)" or profiles["z_m"] != 0
            or profiles["normalization_applied"] or profiles["phase_rotation_applied"]
            or [row["root"] for row in profiles["modes"]] != [1877, 1837]):
        raise ValueError("The original signed mid-thickness profiles differ.")
    metrics = {row["root"]: row for row in section["modes"]}
    y = np.asarray(profiles["coordinates_y_m"], dtype=float)
    if y.shape != (129,) or not np.isfinite(y).all():
        raise ValueError("The profile coordinates differ.")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True, sharey=True)
    scales = {}
    for ax, row in zip(axes, profiles["modes"], strict=True):
        values = np.asarray(row["uz_real"]) + 1j * np.asarray(row["uz_imag"])
        rms = np.sqrt(metrics[row["root"]]["displacement_mass_norm"]
                      / section["grid"]["mass_per_length_kg_m"])
        if values.shape != y.shape or not np.isfinite(values).all() or not np.isfinite(rms) or rms <= 0:
            raise ValueError("The field or full-section RMS is invalid.")
        scales[str(row["root"])] = float(rms)
        ax.plot(y * 1e9, values.real / rms, color="#174a72", linewidth=2, label="Real part")
        ax.plot(y * 1e9, values.imag / rms, color="#bd5c20", linewidth=1.7,
                linestyle="--", label="Imaginary part")
        ax.axhline(0, color="#737d88", linewidth=.7)
        ax.axvline(0, color="#c3c9cf", linewidth=.7)
        ax.set_title(f"Forward root {row['root']}")
        ax.set_xlabel("Width coordinate y (nm)")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel(r"$u_z / u_{\mathrm{rms}}$ (dimensionless)")
    fig.suptitle("Athul Prem | athul_simulation_pnc_one_ghz_mirror\n"
                 "1 GHz forward section fields at z = 0", fontsize=13)
    fig.text(.5, .025, "Each field uses its full-section RMS; original archived complex phase retained. "
             "One component on one line is shown.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .86))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    plt.close(fig)
    print(json.dumps({"author": "Athul Prem", "file_tag": "_AP_pnc", "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output), "source_sha256": sha256(__file__),
        "diagnosis_sha256": sha256(args.diagnosis), "verification_sha256": sha256(args.verification),
        "full_section_rms_m_per_sqrt_w": scales, "display_phase_rotation_applied": False,
        "matplotlib_version": matplotlib.__version__}))


if __name__ == "__main__":
    main()
