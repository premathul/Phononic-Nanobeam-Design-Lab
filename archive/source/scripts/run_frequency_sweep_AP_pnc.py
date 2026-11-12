"""Run PnC and no-hole reference sweeps; only write numerical results from solves."""
from pathlib import Path
import argparse,sys,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
from scripts.solve_harmonic import solve

def run(freqs, mesh, reference_mesh, fields=False):
    rows=[]
    for f in freqs:
        try:
            p, u, sm=solve(mesh,float(f)); r,_,_=solve(reference_mesh,float(f))
            p1=max(p["P1_u"],1e-300); pout=p["P3_u"]
            ref=max(r["P3_u"],1e-300)
            row={"frequency_Hz":f,"frequency_GHz":f/1e9,"P0_u":p["P0_u"],"P1_u":p["P1_u"],"P2_u":p["P2_u"],"P3_u":p["P3_u"],"P4_u":p["P4_u"],"PnC_transmission_amplitude":pout/p1,"PnC_transmission_dB":20*np.log10(max(pout/p1,1e-300)),"reference_output":r["P3_u"],"relative_transmission_dB":20*np.log10(max(pout/ref,1e-300)),"max_displacement":p["max_u"],"solver_status":"PASS"}
            rows.append(row)
            if fields and f in SELECTED_FIELD_FREQUENCIES_HZ:
                np.savez(FIELDS_DIR/f"field_{f/1e9:g}GHz.npz", points=sm.p, displacement=u)
        except Exception as exc:
            rows.append({"frequency_Hz":f,"frequency_GHz":f/1e9,"solver_status":f"FAIL: {type(exc).__name__}: {exc}"})
    out=pd.DataFrame(rows); TABLES_DIR.mkdir(parents=True,exist_ok=True); out.to_csv(TABLES_DIR/"frequency_sweep.csv",index=False); return out

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--test",action="store_true"); ap.add_argument("--full",action="store_true"); a=ap.parse_args()
    freqs=np.array(TEST_FREQUENCIES_HZ if a.test else frequency_list(),float)
    run(freqs,MESH_DIR/"ge_pnc_2d.msh",MESH_DIR/"reference_ge_beam.msh",fields=True)

