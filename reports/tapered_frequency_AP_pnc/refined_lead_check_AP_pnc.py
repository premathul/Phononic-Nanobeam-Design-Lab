"""Check uniform lead channel count at short-listed frequencies. Athul Prem | _AP_pnc."""
import json,sys
from pathlib import Path
import numpy as np
from tools import run_mirror_cell_phase_AP_pnc as cell
from phononic.port_modes_AP_pnc import assemble_port,solve_port_modes
r=[]
for layers in (2,3):
    s=cell._case_spec(100,225,layers)
    body=cell.assemble_case(100,225,layers)
    p=assemble_port(body,s.material,max_dofs=cell.MAX_FACE_DOFS)
    for f in (5.8e9,6e9,6.2e9):
        m=solve_port_modes(p,f); ids=m.positive_indices[m.propagating[m.positive_indices]]
        r.append(dict(frequency_Hz=f,trace_dofs=int(p.basis.N),layer_count=layers,
            right_propagating_count=len(ids),maximum_port_equation_residual=float(np.max(m.relative_residuals)),
            right_propagating_k_rad_m=m.wavenumbers_rad_m[ids].real.tolist(),
            scope="Refined uniform lead check only. Neither finer port is attached to the full device mesh here."))
Path(sys.argv[1]).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
