"""Three local hole-mesh checks for the exact tapered design."""
from pathlib import Path
import csv,sys,gmsh
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import MESH_DIR,FIGURES_DIR,TABLES_DIR
from scripts.solve_direct_fem import prepare,solve_prepared
ROOT=Path(__file__).resolve().parents[1]
holes=[(float(r['position_nm']),float(r['diameter_nm'])) for r in csv.DictReader(open(ROOT/'data/hole_geometry_derived.csv'))]
def build(path,h):
 gmsh.initialize();gmsh.model.add('conv');o=gmsh.model.occ;b=o.addRectangle(-8325,-500,0,16650,1000);hs=[o.addDisk(x,0,0,d/2,d/2) for x,d in holes];c,_=o.cut([(2,b)],[(2,z) for z in hs],removeObject=True,removeTool=True);o.synchronize();cur=[]
 for _,t in gmsh.model.getEntities(1):
  bb=gmsh.model.getBoundingBox(1,t)
  if bb[3]-bb[0]>1e-6 and bb[4]-bb[1]>1e-6:cur.append(t)
 gmsh.model.mesh.field.add('Distance',1);gmsh.model.mesh.field.setNumbers(1,'CurvesList',cur);gmsh.model.mesh.field.add('Threshold',2);gmsh.model.mesh.field.setNumber(2,'InField',1);gmsh.model.mesh.field.setNumber(2,'SizeMin',h);gmsh.model.mesh.field.setNumber(2,'SizeMax',80);gmsh.model.mesh.field.setNumber(2,'DistMin',200);gmsh.model.mesh.field.setNumber(2,'DistMax',1000);gmsh.model.mesh.field.setAsBackgroundMesh(2);gmsh.option.setNumber('Mesh.MeshSizeMin',h);gmsh.option.setNumber('Mesh.MeshSizeMax',80);gmsh.model.mesh.generate(2);gmsh.write(str(path));gmsh.finalize()
rows=[]
for h in (40.,30.,20.):
 path=MESH_DIR/f'convergence_h{int(h)}.msh';build(path,h);p=solve_prepared(prepare(path),5.5e9);rows.append({'hole_target_nm':h,'frequency_GHz':5.5,'nodes':p['n_nodes'],'triangles':p['n_triangles'],'P1_u':p['P1_u'],'P3_u':p['P3_u'],'max_u':p['max_u'],'residual':p['residual']})
with open(TABLES_DIR/'mesh_convergence.csv','w',newline='') as f:
 keys=list(rows[0]);w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
fig,ax=plt.subplots(figsize=(6,4));ax.plot([r['hole_target_nm'] for r in rows],[r['P3_u'] for r in rows],'o-');ax.set(xlabel='target hole mesh size (nm)',ylabel='P3 mean |u| at 5.5 GHz (m)');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIGURES_DIR/'mesh_convergence.png',dpi=220);plt.close(fig);print(rows)
