from pathlib import Path
import sys,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import TABLES_DIR, FIGURES_DIR, SUPPRESSION_THRESHOLDS_DB

def regions(df, threshold):
    ok=df.relative_transmission_dB<=threshold; out=[]; start=None
    for i,v in enumerate(ok):
        if v and start is None: start=i
        if start is not None and (not v or i==len(ok)-1):
            end=i if v and i==len(ok)-1 else i-1; seg=df.iloc[start:end+1]
            out.append(dict(threshold=threshold,lower_frequency_GHz=seg.frequency_GHz.iloc[0],upper_frequency_GHz=seg.frequency_GHz.iloc[-1],bandwidth_GHz=seg.frequency_GHz.iloc[-1]-seg.frequency_GHz.iloc[0],center_frequency_GHz=seg.frequency_GHz.mean(),minimum_transmission_dB=seg.relative_transmission_dB.min())); start=None
    return out
def main():
    df=pd.read_csv(TABLES_DIR/"frequency_sweep.csv"); df=df[df.solver_status=="PASS"].copy()
    allr=[]
    for t in SUPPRESSION_THRESHOLDS_DB: allr.extend(regions(df,t))
    pd.DataFrame(allr,columns=["threshold","lower_frequency_GHz","upper_frequency_GHz","bandwidth_GHz","center_frequency_GHz","minimum_transmission_dB"]).to_csv(TABLES_DIR/"suppression_regions.csv",index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(df.frequency_GHz,df.relative_transmission_dB,label="PnC vs reference"); ax.axhline(-20,color="k",ls="--",lw=.7,label="-20 dB"); ax.set(xlabel="frequency (GHz)",ylabel="relative transmission (dB)"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); FIGURES_DIR.mkdir(parents=True,exist_ok=True); fig.savefig(FIGURES_DIR/"transmission_spectrum.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));
    for c in ["P1_u","P2_u","P3_u","P4_u"]: ax.plot(df.frequency_GHz,df[c],label=c)
    ax.set(xlabel="frequency (GHz)",ylabel="mean displacement (arbitrary units)"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES_DIR/"probe_response.png",dpi=220); plt.close(fig)
if __name__ == "__main__": main()

