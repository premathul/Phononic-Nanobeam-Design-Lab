"""Athul Prem | _AP_pnc. Independent exact-on-the-grid radiation control."""
import json,sys
import numpy as np
from scipy.linalg import solve,eig
from blank_dispersion_AP_pnc import cell,repeat,cat,w
def modes(A):
 n=len(A)//2; scale=np.linalg.norm(A,ord=np.inf); C=A/scale
 aa,bb,cc,dd=C[:n,:n],C[:n,n:],C[n:,:n],C[n:,n:]
 I=np.eye(n); Z=np.zeros((n,n))
 vals,vec=eig(np.block([[Z,I],[-cc,-(aa+dd)]]),np.block([[I,Z],[Z,bb]]))
 U=vec[:n]; V=vec[n:]
 assert np.isfinite(vals).all() and np.min(abs(vals))>0
 axial=-(aa@U+bb@V)*scale
 P=w/2*np.sum(U.conj()*axial,axis=0).imag
 prop=abs(abs(vals)-1)<1e-8
 right=(~prop & (abs(vals)<1))|(prop & (P>0)); left=~right
 assert sum(right)==sum(left)==n
 U[:,prop]/=np.sqrt(abs(P[prop]))
 axial[:,prop]/=np.sqrt(abs(P[prop]))
 ip=np.where(right)[0]; im=np.where(left)[0]
 return U[:,im],U[:,ip],axial[:,im],axial[:,ip],prop[im],prop[ip],vals[ip][prop[ip]]
def solve_blank(A,N,N_transition=None):
 um,up,fm,fp,qm,qp,lam=modes(A)
 X=repeat(A,N); ref=lam**N
 if N_transition is not None:
  B=cell(N_transition,16,2)[2]
  # Same full length: 50% fine, 50% coarse, with a central mesh-only change.
  assert N%4==N_transition%2==0
  X=cat(cat(repeat(A,N//4),repeat(B,N_transition//2)),repeat(A,N//4))
 n=len(A)//2
 zl=solve(um.T,-fm.T).T;zr=solve(up.T,fp.T).T
 Q=X.astype(complex);Q[:n,:n]-=zl;Q[n:,n:]-=zr
 F=np.r_[-fp[:,qp]-zl@up[:,qp],np.zeros((n,sum(qp)))]
 U=solve(Q,F);R=solve(um,U[:n]-up[:,qp])[qm];T=solve(up,U[n:])[qp]
 power=sum(abs(R)**2)+sum(abs(T)**2)
 print(json.dumps(dict(N=N,transition_N=N_transition,mode_count=int(sum(qp)),max_R_pct=float(100*max(sum(abs(R)**2))),max_phase_ref_error=float(np.max(abs(T-np.diag(ref)))),max_flux_error=float(max(abs(1-power))))),flush=True)
if __name__=='__main__':
 N=384; A=cell(N,16,2)[2]
 for t in [None,192,96]:solve_blank(A,N,t)
