"""Direct 2-D plane-stress FEM using Gmsh + NumPy + SciPy.

Coordinates in the Gmsh files are nanometers.  They are converted to metres
before assembly.  The model uses unit out-of-plane thickness, so absolute
displacement amplitudes are source-normalization dependent; ratios are the
reported quantity.
"""
from pathlib import Path
import argparse, csv, math, sys
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from config import MESH_DIR, RESULTS_DIR, FIGURES_DIR, FIELDS_DIR, TABLES_DIR, FREQ_START_HZ, FREQ_STOP_HZ, FREQ_STEP_HZ

RHO=5323.0; E=102.1e9; NU=0.273
BASE_LOSS=0.0
SPONGE_LOSS=2.0e-2
SPONGE_LENGTH_NM=1200.0
TRACTION_PA=1.0
P1_X_NM=-7000.0; P2_X_NM=0.0; P3_X_NM=7000.0; P4_X_NM=8325.0
PROBE_HALF_WIDTH_NM=250.0
PNC_LEFT_NM=-6825.0; PNC_RIGHT_NM=6825.0

def physical_names():
    import gmsh
    return {(int(d),int(t)): gmsh.model.getPhysicalName(d,t) for d,t in gmsh.model.getPhysicalGroups()}

def load_mesh(path):
    import gmsh
    gmsh.initialize(); gmsh.open(str(path))
    tags, coords, _=gmsh.model.mesh.getNodes()
    points=np.asarray(coords,dtype=float).reshape(-1,3)[:,:2]*1e-9
    node_index={int(t):i for i,t in enumerate(tags)}
    tri=[]
    for typ, _, block in zip(*gmsh.model.mesh.getElements(2)):
        prop=gmsh.model.mesh.getElementProperties(typ)
        if prop[0].startswith('Triangle') and int(prop[3])==3:
            tri.extend(np.asarray(block,dtype=np.int64).reshape(-1,3))
    triangles=np.asarray([[node_index[int(n)] for n in row] for row in tri],dtype=np.int64)
    # Unique edges of the triangular domain identify the free exterior boundary.
    counts={}
    for a,b,c in triangles:
        for e in ((a,b),(b,c),(c,a)):
            key=tuple(sorted((int(e[0]),int(e[1])))); counts[key]=counts.get(key,0)+1
    boundary=np.asarray([e for e,n in counts.items() if n==1],dtype=np.int64)
    names=physical_names()
    gmsh.finalize()
    if len(triangles)==0: raise RuntimeError(f'No first-order triangles found in {path}')
    return points,triangles,boundary,names

def constitutive_plane_stress():
    return E/(1-NU**2)*np.array([[1,NU,0],[NU,1,0],[0,0,(1-NU)/2]],float)

def assemble(points, triangles):
    n=len(points); ndof=2*n; D=constitutive_plane_stress()
    rows=[]; cols=[]; kval=[]; mval=[]; cval=[]
    xmin=points[:,0].min()/1e-9; xmax=points[:,0].max()/1e-9
    for conn in triangles:
        xy=points[conn]; x=xy[:,0]; y=xy[:,1]
        det=(x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]); area=abs(det)/2
        if area<=0: raise ValueError('Zero-area element')
        b=np.array([y[1]-y[2],y[2]-y[0],y[0]-y[1]])
        c=np.array([x[2]-x[1],x[0]-x[2],x[1]-x[0]])
        B=np.zeros((3,6)); B[0,0::2]=b; B[1,1::2]=c; B[2,0::2]=c; B[2,1::2]=b; B/=2*area
        ke=B.T@D@B*area
        me=RHO*area/12*np.array([[2,1,1],[1,2,1],[1,1,2]])
        me6=np.zeros((6,6)); me6[0::2,0::2]=me; me6[1::2,1::2]=me
        xc=float(np.mean(xy[:,0])/1e-9)
        d=max(0.0,PNC_LEFT_NM-xc,xc-PNC_RIGHT_NM); profile=min(1.0,(d/SPONGE_LENGTH_NM)**2)
        ce=(BASE_LOSS+SPONGE_LOSS*profile)*2*np.pi*4e9*me6
        dofs=np.array([2*conn[0],2*conn[0]+1,2*conn[1],2*conn[1]+1,2*conn[2],2*conn[2]+1])
        for i,I in enumerate(dofs):
            for j,J in enumerate(dofs):
                rows.append(I); cols.append(J); kval.append(ke[i,j]); mval.append(me6[i,j]); cval.append(ce[i,j])
    shape=(ndof,ndof)
    return csr_matrix((kval,(rows,cols)),shape=shape),csr_matrix((mval,(rows,cols)),shape=shape),csr_matrix((cval,(rows,cols)),shape=shape)

def traction_load(points,boundary):
    xmin=points[:,0].min(); tol=1e-14
    left=[e for e in boundary if abs(points[e[0],0]-xmin)<tol and abs(points[e[1],0]-xmin)<tol]
    if not left: raise RuntimeError('No left boundary edges identified')
    F=np.zeros(2*len(points),complex)
    for a,b in left:
        L=np.linalg.norm(points[a]-points[b]); F[2*a]+=TRACTION_PA*L/2; F[2*b]+=TRACTION_PA*L/2
    return F,left

def probes(points,u):
    out={}
    for name,x0 in [('P1',P1_X_NM),('P2',P2_X_NM),('P3',P3_X_NM),('P4',P4_X_NM)]:
        mask=np.abs(points[:,0]/1e-9-x0)<=PROBE_HALF_WIDTH_NM
        if not np.any(mask): raise RuntimeError(f'No nodes in {name} probe window')
        ux=u[0::2][mask]; uy=u[1::2][mask]
        out[name+'_u']=float(np.mean(np.sqrt(np.abs(ux)**2+np.abs(uy)**2)))
        out[name+'_ux']=float(np.mean(np.abs(ux))); out[name+'_uy']=float(np.mean(np.abs(uy)))
    return out

def prepare(mesh_path):
    points,triangles,boundary,names=load_mesh(mesh_path)
    K,M,C=assemble(points,triangles); F,left=traction_load(points,boundary)
    return points,triangles,K,M,C,F

def solve_prepared(prepared, frequency_hz, field_path=None, field_image=None):
    import matplotlib.pyplot as plt
    points,triangles,K,M,C,F=prepared; w=2*np.pi*frequency_hz
    A=K-(w*w)*M+1j*w*C; u=spsolve(A,F)
    residual=np.linalg.norm(A@u-F)/max(np.linalg.norm(F),1e-300)
    umag=np.sqrt(np.abs(u[0::2])**2+np.abs(u[1::2])**2)
    if not np.all(np.isfinite(u)) or not np.isfinite(residual): raise RuntimeError('NaN/Inf in displacement or residual')
    if float(np.max(umag))<=0: raise RuntimeError('Zero displacement field')
    if residual>1e-7: raise RuntimeError(f'Residual too large: {residual:.3e}')
    p=probes(points,u); p.update({'max_u':float(np.max(umag)),'residual':float(residual),'n_nodes':len(points),'n_triangles':len(triangles),'solver_status':'PASS'})
    if field_path: np.savez(field_path,points_nm=points/1e-9,triangles=triangles,displacement=u)
    if field_image:
        import matplotlib.tri as mtri
        triobj=mtri.Triangulation(points[:,0]/1e-6,points[:,1]/1e-6,triangles)
        fig,ax=plt.subplots(figsize=(14,3.6)); pc=ax.tripcolor(triobj,umag,shading='gouraud',cmap='magma'); fig.colorbar(pc,ax=ax,label='|u| (m)'); ax.set(xlabel='x (µm)',ylabel='y (µm)',aspect='equal'); ax.set_title(f'In-plane displacement magnitude, {frequency_hz/1e9:g} GHz'); fig.tight_layout(); fig.savefig(field_image,dpi=220); plt.close(fig)
    return p

def solve_one(mesh_path, frequency_hz, field_path=None, field_image=None):
    return solve_prepared(prepare(mesh_path), frequency_hz, field_path, field_image)

def generate_reference_mesh(path):
    import gmsh
    gmsh.initialize(); gmsh.model.add('reference_ge_beam'); o=gmsh.model.occ
    xmin=-8325.; xmax=8325.; ymin=-500.; ymax=500.; s=o.addRectangle(xmin,ymin,0,xmax-xmin,ymax-ymin); o.synchronize()
    gmsh.model.addPhysicalGroup(2,[s],1); gmsh.model.setPhysicalName(2,1,'DOMAIN_GE')
    gmsh.option.setNumber('Mesh.MeshSizeMin',80); gmsh.option.setNumber('Mesh.MeshSizeMax',80); gmsh.model.mesh.generate(2); gmsh.write(str(path)); gmsh.finalize()

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def run_frequencies(freqs, pnc_mesh, ref_mesh, save_fields=True):
    pprep=prepare(pnc_mesh); rprep=prepare(ref_mesh)
    rows=[]
    for f in freqs:
        print(f'SOLVE {f/1e9:g} GHz',flush=True)
        field=FIELDS_DIR/f'field_{f/1e9:g}GHz.npz' if save_fields else None
        image=FIGURES_DIR/f'field_{f/1e9:g}GHz.png' if save_fields else None
        p=solve_prepared(pprep,float(f),field,image); r=solve_prepared(rprep,float(f))
        amp=p['P3_u']/max(p['P1_u'],1e-300); rel=20*math.log10(max(p['P3_u'],1e-300)/max(r['P3_u'],1e-300))
        row={'frequency_Hz':f,'frequency_GHz':f/1e9,'P1_u':p['P1_u'],'P2_u':p['P2_u'],'P3_u':p['P3_u'],'P4_u':p['P4_u'],'PnC_transmission_amplitude':amp,'PnC_transmission_dB':20*math.log10(max(amp,1e-300)),'reference_output':r['P3_u'],'relative_transmission_dB':rel,'solver_residual_PnC':p['residual'],'solver_residual_reference':r['residual'],'solver_status':'PASS'}; rows.append(row)
    return rows

def detect(rows):
    out=[]
    for threshold in (-10.,-20.,-30.):
        i=0
        while i<len(rows):
            if rows[i]['relative_transmission_dB']>threshold: i+=1; continue
            j=i
            while j+1<len(rows) and rows[j+1]['relative_transmission_dB']<=threshold: j+=1
            seg=rows[i:j+1]; out.append({'threshold':threshold,'lower_frequency_GHz':seg[0]['frequency_GHz'],'upper_frequency_GHz':seg[-1]['frequency_GHz'],'bandwidth_GHz':seg[-1]['frequency_GHz']-seg[0]['frequency_GHz'],'center_frequency_GHz':sum(x['frequency_GHz'] for x in seg)/len(seg),'minimum_transmission_dB':min(x['relative_transmission_dB'] for x in seg)}); i=j+1
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['4','five','full'],default='4'); a=ap.parse_args(); FIGURES_DIR.mkdir(exist_ok=True,parents=True); TABLES_DIR.mkdir(exist_ok=True,parents=True); FIELDS_DIR.mkdir(exist_ok=True,parents=True)
    pnc=MESH_DIR/'ge_pnc_2d.msh'; ref=MESH_DIR/'reference_ge_beam.msh';
    if not ref.exists(): generate_reference_mesh(ref)
    if a.stage=='4':
        r=solve_one(pnc,4e9,FIELDS_DIR/'field_4GHz.npz',FIGURES_DIR/'field_4GHz.png'); print(r); return
    freqs=np.array([2e9,4e9,6e9,8e9,10e9]) if a.stage=='five' else np.arange(FREQ_START_HZ,FREQ_STOP_HZ+0.5*FREQ_STEP_HZ,FREQ_STEP_HZ)
    rows=run_frequencies(freqs,pnc,ref,save_fields=True); write_csv(TABLES_DIR/('five_frequency_transmission.csv' if a.stage=='five' else 'frequency_sweep.csv'),rows)
    if a.stage=='full':
        write_csv(TABLES_DIR/'suppression_regions.csv',detect(rows)); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4)); ax.plot([r['frequency_GHz'] for r in rows],[r['relative_transmission_dB'] for r in rows]); ax.axhline(-10,color='k',ls=':',lw=.7); ax.axhline(-20,color='k',ls='--',lw=.7); ax.axhline(-30,color='k',ls='-.',lw=.7); ax.set(xlabel='frequency (GHz)',ylabel='relative transmission (dB)'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES_DIR/'transmission_spectrum.png',dpi=220); plt.close(fig)
    print('ROWS',len(rows)); print('MIN_RELATIVE',min(rows,key=lambda x:x['relative_transmission_dB']))

if __name__=='__main__': main()
