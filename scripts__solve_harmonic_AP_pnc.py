"""Frequency-domain plane-stress elasticity solve on a triangular mesh."""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
from scripts.common import probe_definitions

def load_basis(mesh_path):
    try:
        import meshio
        from skfem import MeshTri, Basis, ElementTriP1, ElementVector
    except ImportError as e: raise RuntimeError("meshio and scikit-fem are required for solving") from e
    m=meshio.read(mesh_path); pts=m.points[:,:2]; tri=None
    for cell in m.cells:
        if getattr(cell,"type",None)=="triangle": tri=cell.data; break
    if tri is None and isinstance(m.cells,dict): tri=m.cells["triangle"]
    if tri is None: raise ValueError("No triangular cells in mesh")
    skm=MeshTri(pts.T,tri.T)
    return skm, Basis(skm, ElementVector(ElementTriP1()))

def assemble_system(basis, frequency_hz):
    from skfem import asm, BilinearForm, LinearForm
    from skfem.helpers import ddot, dot, sym_grad, trace
    lam=E*NU/(1-NU**2); mu=E/(2*(1+NU)); omega=2*np.pi*frequency_hz
    @BilinearForm
    def k(u,v,w):
        eu=sym_grad(u); ev=sym_grad(v)
        return 2*mu*ddot(eu,ev)+lam*trace(eu)*trace(ev)
    @BilinearForm
    def mass(u,v,w): return RHO*dot(u,v)
    @LinearForm
    def source(v,w): return F0*v[0]
    # Plane-stress stiffness and consistent mass.  A small complex loss term
    # regularizes finite-domain resonances; it is not an absorbing PML.
    K=asm(k,basis); M=asm(mass,basis)
    A=K-(omega**2)*(1+1j*LOSS_FACTOR)*M
    facets=basis.mesh.facets_satisfying(lambda x: np.isclose(x[0], x[0].min(), rtol=0, atol=1e-15))
    fb=basis.boundary(facets=facets)
    F=asm(source,fb)
    return A,F

def solve(mesh_path, frequency_hz):
    from skfem import condense, solve as sksolve
    skm,basis=load_basis(mesh_path); A,F=assemble_system(basis,frequency_hz)
    # Pin one uy DOF to remove the rigid transverse mode while leaving ux drive free.
    dofs=basis.nodal_dofs; fixed=np.array([dofs[1,0]],dtype=int)
    u=sksolve(*condense(A,F,D=fixed))
    xy=skm.p[:, :]
    vals=np.sqrt(u[0::2]**2+u[1::2]**2)
    measurements={}
    for name,(x0,half) in probe_definitions().items():
        mask=np.abs(xy[0]/NM-x0)<=half
        if not np.any(mask): measurements[name+"_u"]=np.nan; measurements[name+"_ux"]=np.nan; measurements[name+"_uy"]=np.nan
        else:
            measurements[name+"_u"]=float(np.mean(np.abs(vals[mask])))
            measurements[name+"_ux"]=float(np.mean(np.abs(u[0::2][mask])))
            measurements[name+"_uy"]=float(np.mean(np.abs(u[1::2][mask])))
    measurements["max_u"]=float(np.max(np.abs(vals))); measurements["solver_status"]="PASS"
    return measurements,u,skm

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("frequency_hz",type=float); ap.add_argument("--mesh",default=str(MESH_DIR/"ge_pnc_2d.msh")); a=ap.parse_args()
    print(solve(a.mesh,a.frequency_hz)[0])
