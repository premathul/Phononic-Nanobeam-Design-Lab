"""Athul Prem | _AP_pnc. Retarded FE lead impedances by doubling, independent of QZ eigenvectors."""
import json
import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import splu
from blank_dispersion_AP_pnc import repeat,cat,L,w,mat
from discrete_blank_AP_pnc import modes
from phononic.geometry_AP_pnc import BeamGeometry,make_straight_mesh
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.port_modes_AP_pnc import assemble_port
N=384;g=BeamGeometry(L/N,1e-6,1e-7)
ss=assemble_elasticity(make_straight_mesh(g,[1,16,2]),mat)
p=assemble_port(ss,mat,max_dofs=600)
b=np.r_[p.left_dofs,p.right_dofs];i=np.setdiff1d(np.arange(ss.K.shape[0]),b);n=len(b)//2
def dynamic(eta):
 K=(ss.K-w**2*(1+1j*eta)**2*ss.M).tocsc()
 return K[b,:][:,b].toarray()-K[b,:][:,i]@splu(K[i,:][:,i]).solve(K[i,:][:,b].toarray())
A=dynamic(0).real
um,up,fm,fp,qm,qp,lam=modes(A)
X=repeat(A,N)
for eta in [1e-5,1e-6,1e-7,1e-8]:
 Y=dynamic(eta); Y=(Y+Y.T)/2
 for j in range(41):
  Y=cat(Y,Y)
  coupling=float(np.linalg.norm(Y[:n,n:],'fro')/np.linalg.norm(Y[:n,:n],'fro'))
  if coupling<1e-14:break
 assert coupling<1e-14
 zl,zr=-Y[n:,n:],-Y[:n,:n]
 Q=X.astype(complex);Q[:n,:n]-=zl;Q[n:,n:]-=zr
 F=np.r_[-fp[:,qp]-zl@up[:,qp],np.zeros((n,sum(qp)))]
 U=solve(Q,F);Ft=X@U
 def project(bu,bf,du,df,direction):
  return (w/4j)*direction*(bu.conj().T@df-bf.conj().T@du)
 R=project(um[:,qm],fm[:,qm],U[:n]-up[:,qp],-Ft[:n]-fp[:,qp],-1)
 T=project(up[:,qp],fp[:,qp],U[n:],Ft[n:],1)
 print(json.dumps(dict(eta=eta,max_R_pct=float(100*max(sum(abs(R)**2))),max_phase_ref_error=float(np.max(abs(T-np.diag(lam**N)))),impedance_doublings=j+1,remaining_relative_endcoupling=coupling,condition_operator=float(np.linalg.cond(Q)))),flush=True)
