"""Compare unpatterned, periodic-mirror, and tapered-defect finite beams."""
from pathlib import Path
import argparse,csv,math,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import MESH_DIR, TABLES_DIR, FIGURES_DIR, FIELDS_DIR, FREQ_START_HZ, FREQ_STOP_HZ, FREQ_STEP_HZ
from scripts.solve_direct_fem import prepare, solve_prepared

def write(path,rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['five','full'],default='five');a=ap.parse_args()
    freqs=np.array([2e9,4e9,6e9,8e9,10e9]) if a.stage=='five' else np.arange(FREQ_START_HZ,FREQ_STOP_HZ+0.5*FREQ_STEP_HZ,FREQ_STEP_HZ)
    meshes={'unpatterned':MESH_DIR/'unpatterned_ge_beam.msh','periodic_mirror':MESH_DIR/'periodic_mirror_beam.msh','tapered_defect':MESH_DIR/'ge_pnc_2d.msh'}
    prep={k:prepare(v) for k,v in meshes.items()}; rows=[]
    for f in freqs:
        print('COMPARE',f/1e9,'GHz',flush=True); vals={}
        for k in meshes:
            image=(FIGURES_DIR/f'{k}_field_{f/1e9:g}GHz.png') if (a.stage=='five' and k=='tapered_defect') else None
            field=(FIELDS_DIR/f'{k}_field_{f/1e9:g}GHz.npz') if (a.stage=='five' and k=='tapered_defect') else None
            vals[k]=solve_prepared(prep[k],float(f),field,image)
        ref=max(vals['unpatterned']['P3_u'],1e-300)
        rows.append({'frequency_Hz':f,'frequency_GHz':f/1e9,'unpatterned_out':vals['unpatterned']['P3_u'],'periodic_out':vals['periodic_mirror']['P3_u'],'tapered_defect_out':vals['tapered_defect']['P3_u'],'periodic_relative_dB':20*math.log10(max(vals['periodic_mirror']['P3_u'],1e-300)/ref),'defect_relative_dB':20*math.log10(max(vals['tapered_defect']['P3_u'],1e-300)/ref),'periodic_residual':vals['periodic_mirror']['residual'],'defect_residual':vals['tapered_defect']['residual'],'solver_status':'PASS'})
    TABLES_DIR.mkdir(exist_ok=True,parents=True); write(TABLES_DIR/('three_structure_five_frequency.csv' if a.stage=='five' else 'three_structure_frequency_sweep.csv'),rows)
    if a.stage=='full':
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['frequency_GHz'] for r in rows],[r['periodic_relative_dB'] for r in rows],label='periodic mirror / unpatterned');ax.plot([r['frequency_GHz'] for r in rows],[r['defect_relative_dB'] for r in rows],label='tapered defect / unpatterned');ax.axhline(-10,color='k',ls=':',lw=.7);ax.axhline(-20,color='k',ls='--',lw=.7);ax.set(xlabel='frequency (GHz)',ylabel='relative transmission (dB)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIGURES_DIR/'three_structure_transmission.png',dpi=220);plt.close(fig)
    print('rows',len(rows),'periodic_min',min(rows,key=lambda r:r['periodic_relative_dB']),'defect_min',min(rows,key=lambda r:r['defect_relative_dB']))
if __name__=='__main__':main()
