import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from config import WORKBOOK
from scripts.common import read_holes, validate_holes
def data(): return validate_holes(read_holes(WORKBOOK)[0])
def test_workbook_parsing(): assert "Nanobeam Hole Array" in read_holes(WORKBOOK)[1]
def test_26_holes(): assert len(data())==26
def test_position_symmetry():
    d=data(); assert np.allclose(d.position_nm.to_numpy(), -d.position_nm.to_numpy()[::-1])
def test_diameter_range(): assert data().diameter_nm.between(250,275).all()
def test_central_holes(): assert set(data().position_nm.iloc[12:14])=={-250,250}
def test_spacing_and_gaps(): assert (data().validation_status=="PASS").all()

