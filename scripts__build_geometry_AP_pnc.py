"""Build exact 2-D CAD geometry with Gmsh/OpenCASCADE."""
from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import *
from scripts.common import read_holes, validate_holes, beam_extents_nm

def build(reference=False, model_name="ge_pnc_2d"):
    try: import gmsh
    except ImportError as e: raise RuntimeError("gmsh Python module is unavailable; install ge-pnc first") from e
    df,_,_=read_holes(WORKBOOK); df=validate_holes(df)
    # Use nm as the CAD coordinate unit.  Gmsh's OCC tolerance is absolute;
    # constructing a ~10^-8 m model directly can collapse small features.
    # All exported dimensions and mesh sizes remain explicitly in nm.
    xmin,xmax,ymin,ymax=beam_extents_nm(df)
    gmsh.initialize(); gmsh.model.add(model_name); occ=gmsh.model.occ
    beam=occ.addRectangle(xmin,ymin,0,xmax-xmin,ymax-ymin)
    holes=[]
    if not reference:
        for _,r in df.iterrows():
            rad=float(r.diameter_nm)/2
            holes.append(occ.addDisk(float(r.position_nm),0,0,rad,rad))
    cut, _ = occ.cut([(2,beam)], [(2,h) for h in holes], removeObject=True, removeTool=True) if holes else ([(2,beam)], [])
    occ.synchronize()
    domain_tags=[t for d,t in cut if d==2]
    if not domain_tags: raise RuntimeError("CAD boolean produced no 2-D domain")
    gmsh.model.addPhysicalGroup(2,domain_tags,1); gmsh.model.setPhysicalName(2,1,"DOMAIN_GE")
    # Region markers are intentionally explicit provisional markers.  The
    # first prototype has one connected CAD surface, so these are 2-D marker
    # groups sharing the domain; a later partitioned CAD model can refine them.
    for tag,name in [(7,"LEFT_LEAD"),(8,"PNC_REGION"),(9,"RIGHT_LEAD")]:
        gmsh.model.addPhysicalGroup(2,domain_tags,tag); gmsh.model.setPhysicalName(2,tag,name)
    xmin_tags=[]; xmax_tags=[]; top=[]; bottom=[]; hole_curves=[]
    for dim,tag in gmsh.model.getEntities(1):
        bb=gmsh.model.getBoundingBox(dim,tag)
        xspan, yspan = bb[3]-bb[0], bb[4]-bb[1]
        if abs(bb[0]-xmin)<1e-15 and yspan>1e-15 and xspan<1e-15: xmin_tags.append(tag)
        if abs(bb[3]-xmax)<1e-15 and yspan>1e-15 and xspan<1e-15: xmax_tags.append(tag)
        if abs(bb[4]-ymax)<1e-15 and xspan>1e-15 and yspan<1e-15: top.append(tag)
        if abs(bb[1]-ymin)<1e-15 and xspan>1e-15 and yspan<1e-15: bottom.append(tag)
        if xspan>1e-15 and yspan>1e-15: hole_curves.append(tag)
    def pg(tag,name,vals):
        if vals: gmsh.model.addPhysicalGroup(1,vals,tag); gmsh.model.setPhysicalName(1,tag,name)
    pg(2,"LEFT_BOUNDARY",xmin_tags); pg(3,"RIGHT_BOUNDARY",xmax_tags); pg(4,"TOP_BOUNDARY",top); pg(5,"BOTTOM_BOUNDARY",bottom); pg(6,"HOLE_BOUNDARIES",hole_curves)
    if not reference:
        gmsh.model.mesh.field.add("Distance",1); gmsh.model.mesh.field.setNumbers(1,"CurvesList",hole_curves)
        gmsh.model.mesh.field.add("Threshold",2)
        gmsh.model.mesh.field.setNumber(2,"InField",1)
        gmsh.model.mesh.field.setNumber(2,"SizeMin",MESH_HOLE_NM)
        gmsh.model.mesh.field.setNumber(2,"SizeMax",MESH_BULK_NM)
        gmsh.model.mesh.field.setNumber(2,"DistMin",200)
        gmsh.model.mesh.field.setNumber(2,"DistMax",1000)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.write(str(GEOMETRY_DIR/(model_name+".brep")))
    gmsh.write(str(GEOMETRY_DIR/(model_name+".geo_unrolled")))
    gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_HOLE_NM)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_BULK_NM)
    gmsh.finalize()

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--reference",action="store_true"); ap.add_argument("--name",default=None); a=ap.parse_args()
    build(a.reference, a.name or ("reference_ge_beam" if a.reference else "ge_pnc_2d"))
