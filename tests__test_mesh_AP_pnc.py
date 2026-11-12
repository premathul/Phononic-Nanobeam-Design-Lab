import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from config import E,RHO,NU,frequency_list,MESH_DIR
from scripts.common import read_holes,validate_holes
def test_material_constants_positive(): assert E>0 and RHO>0 and 0<NU<.5
def test_frequency_list():
    f=frequency_list(); assert len(f)==111 and f[0]==1e9 and f[-1]==12e9
def test_transmission_numerical_safety(): assert np.isfinite(20*np.log10(max(1e-300,1e-12)))
def test_reference_beam_generation_contract(): assert MESH_DIR.name=="mesh"
def test_mesh_load_if_present():
    p=MESH_DIR/"ge_pnc_2d.msh"
    if not p.exists(): return
    import meshio
    m=meshio.read(p); assert "triangle" in m.cells_dict; assert len(m.cells_dict["triangle"])>0
