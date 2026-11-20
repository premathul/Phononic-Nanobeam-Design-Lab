"""Render the publication figures from saved device results.

Author: Athul Prem | _AP_pnc
"""

import argparse
from pathlib import Path

from plot_device_schematic_AP_pnc import render as draw_geometry
from plot_thin_frequency_traces_AP_pnc import render as draw_thin_frequency
from plot_transmission_refinement_AP_pnc import render as draw_transmission
from plot_weak_conversion_AP_pnc import render as draw_conversion


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "device_AP_pnc",
        help="Directory for vector PDFs, PNG previews and their numerical CSV tables.",
    )
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    for render in (draw_geometry, draw_transmission, draw_conversion, draw_thin_frequency):
        for path in render(output):
            print(path)


if __name__ == "__main__":
    main()
