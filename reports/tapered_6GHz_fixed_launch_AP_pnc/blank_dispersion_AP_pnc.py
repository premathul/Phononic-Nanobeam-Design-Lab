"""Athul Prem | _AP_pnc. Uniform P2 tetrahedral slab condensation and lossless blank."""
import json, sys
import numpy as np
from scipy.linalg import solve,eig
from scipy.sparse.linalg import splu
from phononic.geometry_AP_pnc import BeamGeometry,make_straight_mesh
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port,solve_port_modes
f=6e9; w=2*np.pi*f; L=16650e-9
mat=CubicMaterial(131.5e9,49.4e9,68.4e9,5343)

def cell(nx,ny,nz):
 h=L/nx
 g=BeamGeometry(h,1e-6,1e-7)
 ss=assemble_elasticity(make_straight_mesh(g,[1,ny,nz]),mat,max_dofs=100000)
 p=assemble_port(ss,mat,max_dofs=1800)
 b=np.r_[p.left_dofs,p.right_dofs]; i=np.setdiff1d(np.arange(ss.K.shape[0]),b)
 K=(ss.K-w**2*ss.M).tocsc(); A=K[b,:][:,b].toarray()
 A-=K[b,:][:,i]@splu(K[i,:][:,i]).solve(K[i,:][:,b].toarray())
 return h,p,(A+A.T)/2

def cat(X,Y):
 n=len(X)//2; D=X[n:,n:]+Y[:n,:n]
 a,b=X[:n,:n],X[:n,n:]; c,d=Y[n:,:n],Y[n:,n:]
 H=np.concatenate((b,c),axis=0); E=H@solve(D,H.T,assume_a='sym')
 E[:n,:n]=a-E[:n,:n]; E[n:,n:]=d-E[n:,n:]; E[:n,n:]*=-1;E[n:,:n]*=-1
 return (E+E.T)/2

def repeat(X,n):
 R=None
 while n:
  if n&1:R=X if R is None else cat(R,X)
  n>>=1
  if n:X=cat(X,X)
 return R

def calc(N,ny,nz):
 h,p,A=cell(N,ny,nz); modes=solve_port_modes(p,f); B=repeat(A,N)
 n=len(A)//2
 up,un=modes.displacements[:,modes.positive_indices],modes.displacements[:,modes.negative_indices]
 fp,fn=modes.axial_forces[:,modes.positive_indices],modes.axial_forces[:,modes.negative_indices]
 zp=solve(up.T,fp.T).T; zn=solve(un.T,-fn.T).T
 jj=modes.propagating[modes.positive_indices]; kk=modes.propagating[modes.negative_indices]
 expected=np.exp(1j*modes.wavenumbers_rad_m[modes.positive_indices[jj]]*L)
 Q=B.astype(complex);Q[:n,:n]-=zn;Q[n:,n:]-=zp
 U=solve(Q,np.r_[-fp[:,jj]-zn@up[:,jj],np.zeros((n,sum(jj)))])
 R=solve(un,U[:n]-up[:,jj])[kk]; T=solve(up,U[n:])[jj]
 scale=np.linalg.norm(A,ord=np.inf);C=A/scale
 blocks=(C[:n,:n],C[:n,n:],C[n:,:n],C[n:,n:])
 aa,bb,cc,dd=blocks; I=np.eye(n); Z=np.zeros((n,n))
 vals,vec=eig(np.block([[Z,I],[-cc,-(aa+dd)]]),np.block([[I,Z],[Z,bb]]))
 u=vec[:n,:]; keep=abs(abs(vals)-1)<1e-7
 v=vals[keep]; z=u[:,keep]; Fin=-(aa@z+(bb@z)*v)
 power=np.sum(z.conj()*Fin,axis=0).imag
 kp=np.sort(np.angle(v[power>0])/h);ks=np.sort(modes.wavenumbers_rad_m[modes.positive_indices[jj]].real)
 nxmatch=len(kp)==len(ks) and np.all(kp>=0)
 return dict(N=N,ny=ny,nz=nz,dx_nm=h*1e9,safe_modes=len(ks),discrete_positive_modes=len(kp),Rmax_pct=float(100*np.max(np.sum(abs(R)**2,axis=0))),T_phase_error=float(np.max(abs(T-np.diag(expected)))),max_expected_unwrapped_phase_shift_rad=float(max(abs(kp-ks))*L) if nxmatch else None,max_relative_wavenumber_shift=float(max(abs(kp-ks)/np.maximum(abs(ks),1))) if nxmatch else None,ks=ks.tolist(),kp=kp.tolist())
if __name__=='__main__':
 for a in sys.argv[1:]:
  print(json.dumps(calc(*map(int,a.split(',')))),flush=True)
