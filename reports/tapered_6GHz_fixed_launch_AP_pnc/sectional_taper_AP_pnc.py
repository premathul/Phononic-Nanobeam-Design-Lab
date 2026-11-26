"""Athul Prem | _AP_pnc. Section-wise P2 FE Schur complement for the complete taper."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import splu
from blank_dispersion_AP_pnc import cat,mat,w,L,f
from phononic.geometry_AP_pnc import BeamGeometry,Hole,read_hole_pattern
from phononic.layered_AP_pnc import make_layered_mesh
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.assembly_batched_AP_pnc import assemble_elasticity_batched
from phononic.port_modes_AP_pnc import assemble_port,solve_port_modes
root=Path(__file__).parent
def section(length,local,Dy,Z,H,facets):
 g=BeamGeometry(length*1e-9,1e-6,1e-7)
 holes=() if local is None else (Hole(local[0]*1e-9,local[1]*1e-9,'section'),)
 mesh=make_layered_mesh(g,holes,H*1e-9,port_mesh_size_m=1e-6/Dy,thickness_layers=Z,circle_points=facets,max_port_dofs=1200)
 ss=(assemble_elasticity_batched(mesh,mat,max_dofs=90000,batch_elements=1200) if H<60 else assemble_elasticity(mesh,mat,max_dofs=55000))
 p=assemble_port(ss,mat,max_dofs=1200)
 # Deterministic face basis order shared by every independent section.
 j=np.lexsort((np.arange(p.basis.N)%3,p.basis.doflocs[1],p.basis.doflocs[0]))
 b=np.r_[p.left_dofs[j],p.right_dofs[j]]; i=np.setdiff1d(np.arange(ss.K.shape[0]),b)
 K=(ss.K-w**2*ss.M).tocsc(); A=K[b,:][:,b].toarray()
 I=K[i,:][:,b].tocsc();lu=splu(K[i,:][:,i]); maxres=0.
 for q in range(0,len(b),32):
  inp=I[:,q:q+32].toarray()
  X=lu.solve(inp)
  A[:,q:q+32]-=I.T@X
 sym=np.linalg.norm(A-A.T)/np.linalg.norm(A);assert sym<1e-9
 return (A+A.T)/2,p,j,ss.K.shape[0]
def sequence(H,ny,nz,facets):
 pat=read_hole_pattern('data/geometry/hole_geometry_derived_AP_pnc.csv')
 centers=np.array([round(h.x_m*1e9+8325,8) for h in pat.holes])
 diams=np.array([round(h.diameter_m*1e9,8) for h in pat.holes])
 limits=np.r_[0.,1000.,(centers[1:]+centers[:-1])/2,15650.,16650.]
 assert len(limits)==29
 if H<=33:
  # Split only the long lead blocks and the hole-free pieces of the outer cells.
  # No added plane intersects a hole; the physical 26-hole layout is unchanged.
  limits=np.sort(np.r_[limits,500.,1350.,15300.,16150.])
 cache={};p=j=None
 for taper in [False,True]:
  AA=None
  for i,(lo,hi) in enumerate(zip(limits[:-1],limits[1:])):
   im=np.flatnonzero((centers>lo)&(centers<hi))
   assert len(im)<=1
   loc=(round(centers[im[0]]-lo,8),diams[im[0]]) if taper and len(im) else None
   if loc is not None:assert loc[1]/2<loc[0]<hi-lo-loc[1]/2
   key=(round(hi-lo,8),loc)
   if key not in cache:
    BB,pp,jj,nd=section(*key,ny,nz,H,facets)
    if p is None:p,j=pp,jj
    else:
     assert np.allclose(pp.basis.doflocs[:,jj],p.basis.doflocs[:,j],rtol=0,atol=1e-16)
     assert np.allclose(pp.M[jj,:][:,jj],p.M[j,:][:,j],rtol=1e-10,atol=0)
    cache[key]=BB
    print('section',H,ny,nz,taper,key,nd,flush=True)
   AA=cache[key] if AA is None else cat(AA,cache[key])
  mode=solve_port_modes(p,f)
  un,up=[mode.displacements[j,:][:,i] for i in (mode.negative_indices,mode.positive_indices)]
  fn,fp=[mode.axial_forces[j,:][:,i] for i in (mode.negative_indices,mode.positive_indices)]
  qn,qp=[mode.propagating[i] for i in (mode.negative_indices,mode.positive_indices)]
  k=mode.wavenumbers_rad_m[mode.positive_indices[qp]]
  n=len(AA)//2; zL=solve(un.T,-fn.T).T;zR=solve(up.T,fp.T).T
  Q=AA.astype(complex);Q[:n,:n]-=zL;Q[n:,n:]-=zR
  rhs=np.zeros((2*n,sum(qp)+sum(qn)),complex)
  rhs[:n,:sum(qp)]=-fp[:,qp]-zL@up[:,qp];rhs[n:,sum(qp):]=fn[:,qn]-zR@un[:,qn]
  U=solve(Q,rhs); Fres=AA@U
  UU=U.copy();UU[:n,:sum(qp)]-=up[:,qp];UU[n:,sum(qp):]-=un[:,qn]
  S=np.r_[solve(un,UU[:n])[qn],solve(up,UU[n:])[qp]]
  assert sum(qp)==sum(qn);m=sum(qp);R,T=S[:m,:m],S[m:,:m]
  expected=np.diag(np.exp(1j*k*L))
  stat=dict(H=H,ny=ny,nz=nz,facets=facets,taper=taper,m=int(m),worst_R_pct=float(100*np.max(np.sum(abs(R)**2,axis=0))),max_coherent_R_pct=float(100*np.linalg.norm(R,2)**2),max_phase_ref_error=float(np.max(abs(T-expected))),unia=float(np.linalg.norm(S.conj().T@S-np.eye(len(S)),2)),T_power_eig=np.linalg.eigvalsh(T.conj().T@T).tolist(),condensate_residual=float(np.linalg.norm(Q@U-rhs)/np.linalg.norm(rhs)))
  print(json.dumps(stat),flush=True)
  prefix=root/f'section_H{H}_ny{ny}_nz{nz}_rim{facets}_{"taper" if taper else "blank"}_AP_pnc'
  np.savez_compressed(str(prefix)+'.npz',S=S,k=k,up=up[:,qp],un=un[:,qn],port_mass=p.M[j,:][:,j],portloc=p.basis.doflocs[:,j],D=AA,section_limits_nm=limits,hole_centers_nm=centers,hole_diameters_nm=diams)
  prefix.with_suffix('.json').write_text(json.dumps(stat,indent=2))
if __name__=='__main__':
 for a in sys.argv[1:]:sequence(*map(int,a.split(',')))
