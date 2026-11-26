"""Athul Prem | _AP_pnc. Re-evaluate saved assembled sections with retarded discrete leads."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.linalg import solve,polar
from scipy.sparse.linalg import splu
from blank_dispersion_AP_pnc import cat,L,w,mat
from discrete_blank_AP_pnc import modes
from phononic.geometry_AP_pnc import BeamGeometry,make_straight_mesh
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.port_modes_AP_pnc import assemble_port
root=Path(__file__).parent
def calc(ny,nz,N,files):
 eta=1e-8
 ss=assemble_elasticity(make_straight_mesh(BeamGeometry(L/N,1e-6,1e-7),[1,ny,nz]),mat)
 p=assemble_port(ss,mat,max_dofs=1200);j=np.lexsort((np.arange(p.basis.N)%3,p.basis.doflocs[1],p.basis.doflocs[0]))
 b=np.r_[p.left_dofs[j],p.right_dofs[j]]; i=np.setdiff1d(np.arange(ss.K.shape[0]),b); n=len(b)//2
 def dyn(eta):
  K=(ss.K-w*w*(1+1j*eta)**2*ss.M).tocsc()
  A=K[b,:][:,b].toarray()-K[b,:][:,i]@splu(K[i,:][:,i]).solve(K[i,:][:,b].toarray())
  return (A+A.T)/2
 A=dyn(0).real
 Y=dyn(eta)
 for it in range(43):
  Y=cat(Y,Y);cpl=float(np.linalg.norm(Y[:n,n:])/np.linalg.norm(Y[:n,:n]))
  if cpl<1e-14:break
 assert cpl<1e-14
 zl,zr=-Y[n:,n:],-Y[:n,:n]
 um,up,fm,fp,qm,qp,lam=modes(A)
 m=sum(qp);idx=np.argsort(np.angle(lam));upp,fpp=up[:,qp][:,idx],fp[:,qp][:,idx]
 # Use conjugated partner basis for reflected waves, preserving phase gauge and reciprocity.
 umm,fmm=upp.conj(),fpp.conj()
 F=np.zeros((2*n,2*m),complex)
 F[:n,:m]=-fpp-zl@upp;F[n:,m:]=fmm-zr@umm
 def project(bu,bf,du,df,sign):return w*sign/(4j)*(bu.conj().T@df-bf.conj().T@du)
 for path in files:
  with np.load(path) as t:
   assert np.array_equal(t['portloc'],p.basis.doflocs[:,j])
   assert np.array_equal(t['port_mass'],p.M[j,:][:,j])
   C=t['D']
  Q=C.astype(complex);Q[:n,:n]-=zl;Q[n:,n:]-=zr
  U=solve(Q,F);Ft=C@U
  UL,FL=U[:n].copy(),-Ft[:n].copy();UL[:,:m]-=upp;FL[:,:m]-=fpp
  UR,FR=U[n:].copy(),Ft[n:].copy();UR[:,m:]-=umm;FR[:,m:]-=fmm
  S=np.r_[project(umm,fmm,UL,FL,-1),project(upp,fpp,UR,FR,1)]
  R,T=S[:m,:m],S[m:,:m]
  d=dict(case=path.name,lead_step_nm=L/N*1e9,mode_count=int(m),worst_R_pct=float(100*np.max(np.sum(abs(R)**2,axis=0))),max_coherent_R_pct=float(100*np.linalg.norm(R,2)**2),max_absolute_phase_error_vs_discrete_uniform=float(np.max(abs(T-np.diag(lam[idx]**N)))),balance=float(np.linalg.norm(S.conj().T@S-np.eye(2*m),2)),symmetric_reciprocity=float(np.linalg.norm(S-S.T,2)),T_eigs=np.linalg.eigvalsh(T.conj().T@T).tolist())
  out=root/f'retarded_dxN{N}_{path.stem}'
  np.savez_compressed(out.with_suffix('.npz'),S=S,incident_u=upp,incident_f=fpp,lambda_positive=lam[idx])
  out.with_suffix('.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
if __name__=='__main__':
 for arg in sys.argv[1:]:
  ny,nz,N=map(int,arg.split(','))
  files=sorted(root.glob(f'section_H*_ny{ny}_nz{nz}_rim*_AP_pnc.npz'))
  files=[x for x in files if not (root/f'retarded_dxN{N}_{x.name}').exists()]
  if files:calc(ny,nz,N,files)
