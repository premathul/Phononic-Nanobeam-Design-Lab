from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WORKBOOK, TABLES_DIR
from scripts.common import read_holes, validate_holes

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df, sheets, columns = read_holes(WORKBOOK)
    print("sheet names:", sheets)
    print("exact columns:", columns)
    validated = validate_holes(df)
    print("valid hole rows:", len(validated))
    print("position range (nm):", validated.position_nm.min(), validated.position_nm.max())
    print("diameter range (nm):", validated.diameter_nm.min(), validated.diameter_nm.max())
    print("regions:", sorted(validated.region.unique()))
    validated.to_csv(TABLES_DIR / "hole_validation.csv", index=False)
    return validated

if __name__ == "__main__": main()

