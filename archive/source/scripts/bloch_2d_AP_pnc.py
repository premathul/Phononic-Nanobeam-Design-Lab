"""Reduced 2-D Bloch band structure for a 550 nm periodic Ge mirror cell."""
from pathlib import Path
import csv,sys
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import MESH_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR
from scripts.solve_direct_fem import load_mesh, assemble, constitutive_plane_stress

A_NM=550.0; WIDTH_NM=1000.; HOLE_D_NM=250.; NBRANCH=16

def make_cell(path):
    import gmsh
    gmsh.initialize(); gmsh.model.add('mirror_unit_cell'); o=gmsh.model.occ
    b=o.addRectangle(-A_NM/2,-WIDTH_NM/2,0,A_NM,WIDTH_NM); h=o.addDisk(0,0,0,HOLE_D_NM/2,HOLE_D_NM/2); cut,_=o.cut([(2,b)],[(2,h)],removeObject=True,removeTool=True);o.synchronize()
    dom=[t for d,t in cut if d==2];gmsh.model.addPhysicalGroup(2,dom,1);gmsh.model.setPhysicalName(2,1,'DOMAIN_GE')
    curves=[]
    for _,t in gmsh.model.getEntities(1):
        bb=gmsh.model.getBoundingBox(1,t)
        if bb[3]-bb[0]>1e-6 and bb[4]-bb[1]<1e-6: curves.append(t)
    gmsh.option.setNumber('Mesh.MeshSizeMin',30);gmsh.option.setNumber('Mesh.MeshSizeMax',80);gmsh.model.mesh.generate(2);MESH_DIR.mkdir(exist_ok=True);gmsh.write(str(path));gmsh.finalize()

def periodic_transform(points):
    x=points[:,0]/1e-9;y=points[:,1]/1e-9; xmin=x.min();xmax=x.max(); tol=1e-6
    L=np.where(np.abs(x-xmin)<tol)[0]; R=np.where(np.abs(x-xmax)<tol)[0]
    pairs=[]
    for i in L:
        j=R[np.argmin(np.abs(y[R]-y[i]))]
        if abs(y[j]-y[i])>1e-3: raise RuntimeError('Unit-cell left/right boundary nodes are not conforming')
        pairs.append((int(i),int(j)))
    return pairs

def reduced_matrices(K,M,points):
    pairs=periodic_transform(points); n=len(points); full=2*n; rep=np.arange(full); pairmap={r:l for l,r in pairs}
    for l,r in pairs: rep[2*r]=2*l;rep[2*r+1]=2*l+1
    used=np.unique(rep); col={int(v):i for i,v in enumerate(used)}
    from scipy.sparse import coo_matrix
    rr=np.arange(full); cc=np.array([col[int(v)] for v in rep]); data=np.ones(full,complex)
    # Right-boundary displacement equals exp(i*k*a) times left-boundary displacement.
    flags=np.ones(full,complex)
    for l,r in pairs: flags[2*r]=np.nan;flags[2*r+1]=np.nan
    return used,rep,col,pairs

def matrices_at_k(K,M,points,k):
    pairs=periodic_transform(points);n=len(points);xmin=points[:,0].min(); full=2*n
    rep=np.arange(full); phase=np.ones(full,complex)
    for l,r in pairs:
        rep[2*r]=rep[2*l];rep[2*r+1]=rep[2*l+1]; ph=np.exp(1j*k*A_NM*1e-9);phase[2*r]=ph;phase[2*r+1]=ph
    used=np.unique(rep); cmap={int(v):i for i,v in enumerate(used)}
    rows=np.arange(full); cols=np.array([cmap[int(v)] for v in rep]); from scipy.sparse import coo_matrix
    T=coo_matrix((phase,(rows,cols)),shape=(full,len(used))).tocsr()
    return (T.conj().T@K@T).toarray(),(T.conj().T@M@T).toarray()

def main():
    path=MESH_DIR/'mirror_unit_cell.msh';
    if not path.exists(): make_cell(path)
    points,tri,boundary,names=load_mesh(path);K,M,C=assemble(points,tri)
    ks=np.linspace(0,np.pi/(A_NM*1e-9),21); bands=[]
    for k in ks:
        kr,mr=matrices_at_k(K,M,points,k); vals=eigh(kr,mr,subset_by_index=[0,min(NBRANCH-1,kr.shape[0]-1)],check_finite=False,eigvals_only=True); bands.append(np.sqrt(np.maximum(vals,0))/(2*np.pi))
    bands=np.asarray(bands); FIGURES_DIR.mkdir(exist_ok=True,parents=True);TABLES_DIR.mkdir(exist_ok=True,parents=True)
    fig,ax=plt.subplots(figsize=(8,5));q=np.arange(bands.shape[1]);
    for j in q: ax.plot(ks/(np.pi/(A_NM*1e-9)),bands[:,j]/1e9,'o-',ms=2,lw=.8,color='#245f9e')
    ax.set(xlabel='k / (π/a)',ylabel='frequency (GHz)',title='2-D in-plane elastic Bloch bands, 550 nm mirror cell');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIGURES_DIR/'band_structure_2d.png',dpi=220);plt.close(fig)
    gaps=[]
    for j in range(bands.shape[1]-1):
        lo=float(np.max(bands[:,j])/1e9);hi=float(np.min(bands[:,j+1])/1e9)
        if hi>lo: gaps.append({'branch_lower':j,'lower_band_edge_GHz':lo,'upper_band_edge_GHz':hi,'center_GHz':(lo+hi)/2,'gap_width_GHz':hi-lo,'normalized_gap_width':(hi-lo)/((hi+lo)/2)})
    with open(TABLES_DIR/'band_gaps_2d.csv','w',newline='') as f:
        keys=['branch_lower','lower_band_edge_GHz','upper_band_edge_GHz','center_GHz','gap_width_GHz','normalized_gap_width'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(gaps)
    print('BLOCH_PASS',len(points),'nodes',bands.shape,'bands','gaps',gaps)
if __name__=='__main__':main()
