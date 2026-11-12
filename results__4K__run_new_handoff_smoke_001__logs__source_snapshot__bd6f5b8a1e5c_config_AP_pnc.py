"""Reproducible configuration for the first-pass Ge PnC simulator.

All geometric dimensions are entered in nm for readability and converted to SI
units in the geometry/solver scripts.  Beam dimensions below are provisional
simulation assumptions; they are not supplied by the workbook.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
GEOMETRY_DIR = PROJECT_ROOT / "geometry"
MESH_DIR = PROJECT_ROOT / "mesh"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIELDS_DIR = RESULTS_DIR / "fields"
TABLES_DIR = RESULTS_DIR / "tables"
WORKBOOK = DATA_DIR / "nanobeam_hole_array.xlsx"

NM = 1e-9
BEAM_WIDTH_NM = 1000.0                 # PROVISIONAL — not experimentally validated
LEFT_LEAD_NM = 1500.0                  # PROVISIONAL — not experimentally validated
RIGHT_LEAD_NM = 1500.0                 # PROVISIONAL — not experimentally validated

MESH_HOLE_NM = 30.0
MESH_BULK_NM = 80.0

RHO = 5323.0                            # kg/m^3, isotropic approximation
E = 102.1e9                             # Pa, isotropic approximation
NU = 0.273                              # Poisson ratio
LOSS_FACTOR = 1e-3                      # dimensionless material-loss regularizer
F0 = 1.0                                # normalized traction amplitude

FREQ_START_HZ = 1e9
FREQ_STOP_HZ = 12e9
FREQ_STEP_HZ = 100e6
TEST_FREQUENCIES_HZ = [2e9, 4e9, 6e9, 8e9, 10e9]
SELECTED_FIELD_FREQUENCIES_HZ = TEST_FREQUENCIES_HZ

P0_X_NM = -BEAM_WIDTH_NM / 2 - LEFT_LEAD_NM * 0.55
P1_X_NM = -6700.0 - 450.0
P2_X_NM = 0.0
P3_X_NM = 6700.0 + 450.0
P4_X_NM = BEAM_WIDTH_NM / 2 + LEFT_LEAD_NM * 0.55
PROBE_HALF_WIDTH_NM = 100.0

SUPPRESSION_THRESHOLDS_DB = [-10.0, -20.0, -30.0]

def frequency_list(start=FREQ_START_HZ, stop=FREQ_STOP_HZ, step=FREQ_STEP_HZ):
    """Deterministic inclusive frequency list."""
    import numpy as np
    return np.arange(start, stop + 0.5 * step, step, dtype=float)

