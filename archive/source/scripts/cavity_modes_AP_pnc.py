"""Finite exact-defect eigenmode search using the direct FEM K/M matrices."""
from pathlib import Path
import csv,sys
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import MESH_DIR, FIGURES_DIR, TABLES_DIR
from scripts.solve_direct_fem import load_mesh, assemble

TARGET_HZ=2.883e9; SEARCH_LOW_HZ=2.830e9; SEARCH_HIGH_HZ=2.936e9
def main():
    pts,tri,boundary,names=load_mesh(MESH_DIR/'ge_pnc_2d.msh'); K,M,C=assemble(pts,tri)
    vals,vecs=eigsh(K,M=M,k=12,sigma=(2*np.pi*TARGET_HZ)**2,which='LM'); order=np.argsort(vals); vals=vals[order];vecs=vecs[:,order]
    rows=[]; selected=[]
    D=102.1e9/(1-0.273**2)*np.array([[1,.273,0],[.273,1,0],[0,0,(1-.273)/2]])
    for j,val in enumerate(vals):
        if val<=0: continue
        f=np.sqrt(val)/(2*np.pi); u=vecs[:,j]; um=np.sqrt(abs(u[0::2])**2+abs(u[1::2])**2)
        center=np.abs(pts[tri].mean(axis=1)[:,0]/1e-9)<2500
        q=np.mean(um[tri]**2,axis=1); area=[]; strains=[]
        for conn in tri:
            xy=pts[conn]; x=xy[:,0];y=xy[:,1]; A=abs((x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]))/2
            b=np.array([y[1]-y[2],y[2]-y[0],y[0]-y[1]]);c=np.array([x[2]-x[1],x[0]-x[2],x[1]-x[0]]);B=np.zeros((3,6));B[0,0::2]=b;B[1,1::2]=c;B[2,0::2]=c;B[2,1::2]=b;B/=2*A; dof=[2*conn[0],2*conn[0]+1,2*conn[1],2*conn[1]+1,2*conn[2],2*conn[2]+1]; strains.append(np.linalg.norm(B@u[dof]));area.append(A)
        area=np.asarray(area); strain=np.asarray(strains); total=np.sum(q*area); loc=np.sum(q[center]*area[center])/max(total,1e-300); eff=total**2/max(np.sum((q*q)*area),1e-300)
        imax=int(np.argmax(strain)); sc=pts[tri].mean(axis=1)[imax]/1e-6
        rows.append({'mode_index':j,'frequency_GHz':f/1e9,'localization_factor_|x|<2.5um':loc,'effective_mode_area_um2':eff/1e-12,'max_displacement_normalized':float(um.max()),'max_strain_normalized':float(strain.max()),'max_strain_x_um':float(sc[0]),'max_strain_y_um':float(sc[1])})
        if SEARCH_LOW_HZ<=f<=SEARCH_HIGH_HZ: selected.append((loc,j,f,u,strain,um))
    if not selected: raise RuntimeError('No finite eigenmode found in the computed 2D Bloch-gap search interval')
    loc,j,f,u,strain,um=max(selected,key=lambda z:z[0]); FIGURES_DIR.mkdir(exist_ok=True,parents=True);TABLES_DIR.mkdir(exist_ok=True,parents=True)
    with open(TABLES_DIR/'cavity_mode_summary.csv','w',newline='') as fp:
        keys=list(rows[0].keys()); w=csv.DictWriter(fp,fieldnames=keys);w.writeheader();w.writerows(rows)
    t=mtri.Triangulation(pts[:,0]/1e-6,pts[:,1]/1e-6,tri)
    fig,ax=plt.subplots(figsize=(14,3.6)); pc=ax.tripcolor(t,um,shading='gouraud',cmap='magma');fig.colorbar(pc,ax=ax,label='normalized |u|');ax.set(xlabel='x (um)',ylabel='y (um)',aspect='equal',title=f'Exact tapered cavity eigenmode near {f/1e9:.3f} GHz');fig.tight_layout();fig.savefig(FIGURES_DIR/'cavity_mode.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,3.6));pc=ax.tripcolor(t,strain,shading='flat',cmap='viridis');fig.colorbar(pc,ax=ax,label='normalized strain magnitude');ax.set(xlabel='x (um)',ylabel='y (um)',aspect='equal',title=f'Cavity eigenmode strain field near {f/1e9:.3f} GHz');fig.tight_layout();fig.savefig(FIGURES_DIR/'cavity_strain.png',dpi=220);plt.close(fig)
    print('CAVITY_PASS',f/1e9,'GHz','localization',loc,'effective_area_um2',eff/1e-12,'mode',j)
if __name__=='__main__':main()
