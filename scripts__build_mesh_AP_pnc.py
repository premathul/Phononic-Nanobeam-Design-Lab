"""Mesh CAD geometry with Gmsh and export MSH/VTK."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
def mesh(name="ge_pnc_2d"):
    try: import gmsh, meshio
    except ImportError as e: raise RuntimeError("gmsh and meshio are required for meshing") from e
    geo=GEOMETRY_DIR/(name+".brep"); msh=MESH_DIR/(name+".msh")
    gmsh.initialize(); gmsh.open(str(GEOMETRY_DIR/(name+".geo_unrolled")))
    gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_HOLE_NM); gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_BULK_NM)
    gmsh.option.setNumber("Mesh.Algorithm", 6); gmsh.model.mesh.generate(2); gmsh.write(str(msh)); gmsh.finalize()
    meshio.write(MESH_DIR/(name+".vtk"), meshio.read(msh))
if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--name",default="ge_pnc_2d"); mesh(ap.parse_args().name)
