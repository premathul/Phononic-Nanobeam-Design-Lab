"""Build reduced-order comparison structures with Gmsh in nanometer units."""
from pathlib import Path
import sys, gmsh
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from config import MESH_DIR, GEOMETRY_DIR

XMIN=-8325.; XMAX=8325.; YMIN=-500.; YMAX=500.
def build(name, holes):
    gmsh.initialize(); gmsh.model.add(name); o=gmsh.model.occ
    beam=o.addRectangle(XMIN,YMIN,0,XMAX-XMIN,YMAX-YMIN)
    hs=[o.addDisk(float(x),0,0,float(d)/2,float(d)/2) for x,d in holes]
    cut,_=o.cut([(2,beam)],[(2,h) for h in hs],removeObject=True,removeTool=True) if hs else ([(2,beam)],[])
    o.synchronize(); dom=[t for k,t in cut if k==2]
    gmsh.model.addPhysicalGroup(2,dom,1); gmsh.model.setPhysicalName(2,1,'DOMAIN_GE')
    gmsh.option.setNumber('Mesh.MeshSizeMin',30 if hs else 80); gmsh.option.setNumber('Mesh.MeshSizeMax',80)
    if hs:
        curves=[]
        for _,t in gmsh.model.getEntities(1):
            bb=gmsh.model.getBoundingBox(1,t)
            if bb[3]-bb[0]>1e-6 and bb[4]-bb[1]>1e-6: curves.append(t)
        gmsh.model.mesh.field.add('Distance',1); gmsh.model.mesh.field.setNumbers(1,'CurvesList',curves)
        gmsh.model.mesh.field.add('Threshold',2); gmsh.model.mesh.field.setNumber(2,'InField',1); gmsh.model.mesh.field.setNumber(2,'SizeMin',30); gmsh.model.mesh.field.setNumber(2,'SizeMax',80); gmsh.model.mesh.field.setNumber(2,'DistMin',200); gmsh.model.mesh.field.setNumber(2,'DistMax',1000); gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(2); GEOMETRY_DIR.mkdir(exist_ok=True); MESH_DIR.mkdir(exist_ok=True)
    gmsh.write(str(GEOMETRY_DIR/f'{name}.brep')); gmsh.write(str(MESH_DIR/f'{name}.msh')); gmsh.finalize()

if __name__=='__main__':
    # One centered 550 nm lattice with no defect: 26 equal 250 nm holes.
    periodic=[((i+0.5)*550.0,250.0) for i in range(-13,13)]
    build('periodic_mirror_beam',periodic)
    build('unpatterned_ge_beam',[])
