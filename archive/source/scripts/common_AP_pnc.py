"""Shared workbook, geometry, and result helpers."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

EXPECTED_SHEET = "Nanobeam Hole Array"

def read_holes(workbook):
    workbook = Path(workbook)
    xls = pd.ExcelFile(workbook)
    if EXPECTED_SHEET not in xls.sheet_names:
        raise ValueError(f"Missing required sheet {EXPECTED_SHEET!r}; found {xls.sheet_names}")
    df = pd.read_excel(workbook, sheet_name=EXPECTED_SHEET)
    df = df.dropna(how="all").copy()
    df = df[df["Hole #"].notna()].copy()
    # The source workbook includes a trailing explanatory note in the Hole #
    # column.  Keep the note in the original workbook, but exclude it from the
    # hole table by requiring a numeric hole identifier.
    numeric_hole = pd.to_numeric(df["Hole #"], errors="coerce")
    df = df[numeric_hole.notna()].copy()
    cols = list(df.columns)
    required = ["Hole #", "Position (nm)", "Diameter (nm)", "Region"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; exact columns: {cols}")
    spacing = next(c for c in cols if c.startswith("Center-to-center distance"))
    gap = next(c for c in cols if c.startswith("Edge-to-edge gap"))
    out = pd.DataFrame({
        "hole_number": pd.to_numeric(df["Hole #"], errors="raise").astype(int),
        "position_nm": pd.to_numeric(df["Position (nm)"], errors="raise"),
        "diameter_nm": pd.to_numeric(df["Diameter (nm)"], errors="raise"),
        "spreadsheet_spacing_nm": df[spacing].map(parse_number),
        "spreadsheet_gap_nm": df[gap].map(parse_number),
        "region": df["Region"].astype(str).str.strip(),
    })
    return out, xls.sheet_names, cols

def parse_number(value):
    if pd.isna(value):
        return np.nan
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(m.group()) if m else np.nan

def validate_holes(df, tolerance_nm=1e-9):
    if len(df) != 26:
        raise ValueError(f"Expected exactly 26 holes, found {len(df)}")
    if not np.all(np.diff(df.position_nm) > 0):
        raise ValueError("Hole positions must be strictly increasing")
    calculated_spacing = df.position_nm.diff()
    radii = df.diameter_nm / 2.0
    calculated_gap = calculated_spacing - radii.shift(1) - radii
    status = []
    for i in range(len(df)):
        ok = True
        if i:
            ok = (abs(calculated_spacing.iloc[i] - df.spreadsheet_spacing_nm.iloc[i]) <= tolerance_nm and
                  abs(calculated_gap.iloc[i] - df.spreadsheet_gap_nm.iloc[i]) <= tolerance_nm)
        status.append("PASS" if ok else "FAIL")
    result = df.copy()
    result["radius_nm"] = radii
    result["calculated_spacing_nm"] = calculated_spacing
    result["calculated_gap_nm"] = calculated_gap
    result["validation_status"] = status
    if any(s != "PASS" for s in status):
        raise ValueError("Workbook spacing/gap validation failed; see generated validation table")
    return result

def beam_extents_nm(df):
    from config import BEAM_WIDTH_NM, LEFT_LEAD_NM, RIGHT_LEAD_NM
    return (float(df.position_nm.min() - df.diameter_nm.iloc[0] / 2 - LEFT_LEAD_NM),
            float(df.position_nm.max() + df.diameter_nm.iloc[-1] / 2 + RIGHT_LEAD_NM),
            -BEAM_WIDTH_NM / 2, BEAM_WIDTH_NM / 2)

def probe_definitions():
    from config import P0_X_NM, P1_X_NM, P2_X_NM, P3_X_NM, P4_X_NM, PROBE_HALF_WIDTH_NM
    return {k: (x, PROBE_HALF_WIDTH_NM) for k, x in zip("P0 P1 P2 P3 P4".split(), [P0_X_NM, P1_X_NM, P2_X_NM, P3_X_NM, P4_X_NM])}
