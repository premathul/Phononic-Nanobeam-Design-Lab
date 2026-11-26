"""Athul Prem | _AP_pnc. Freeze launch subspaces before testing on independently refined meshes."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import qr
from blank_dispersion_AP_pnc import cell
root=Path(__file__).parent
base=[root/f'section_H{h}_ny16_nz2_rim32_taper_AP_pnc.npz' for h in [85,60]]
def power(t):
 m=len(t['k']);T=t['S'][m:,:m];return T.conj().T@T
def data(p):
 with np.load(p,allow_pickle=False) as f:return dict(f)
aa,bb=map(data,base)
assert np.array_equal(aa['up'],bb['up'])
g1,g2=map(power,(aa,bb))
z,low=np.linalg.eigh(g1+g2); zl,high=np.linalg.eigh(2*np.eye(len(z))-g1-g2)
states=[('rejected',low[:,:6]),('passed',high[:,:1])]
nodes,weights=np.polynomial.legendre.leggauss(5)
def pts(N,lo,hi):
 h=(hi-lo)/N
 return (lo+h*(np.arange(N)[:,None]+(nodes[None,:]+1)/2)).ravel(),np.tile(weights*h/2,N)
y,wy=pts(80,-.5e-6,.5e-6);z,wz=pts(12,-.5e-7,.5e-7)
X=np.array(np.meshgrid(y,z,indexing='ij')).reshape(2,-1)
wt=np.sqrt(wy[:,None]*wz).ravel()
def shape(t,ny,nz):
 _,p,_=cell(384,ny,nz)
 j=np.lexsort((np.arange(p.basis.N)%3,p.basis.doflocs[1],p.basis.doflocs[0]))
 assert np.array_equal(t['portloc'],p.basis.doflocs[:,j]) and np.array_equal(t['port_mass'],p.M[j,:][:,j])
 out=np.zeros_like(t['up']);out[j,:]=t['up']
 return np.vstack([(bas.probes(X)@out[ids,:])*wt[:,None] for ids,bas in zip(p.basis.split_indices(),p.basis.split_bases())])
ref=shape(aa,16,2)
if __name__=='__main__':
 for p in sorted(root.glob('section_H*_taper_AP_pnc.npz')):
  ny=int(p.name.split('_ny')[1].split('_')[0]);nz=int(p.name.split('_nz')[1].split('_')[0])
  t=data(p); g=power(t); qb=power(data(str(p).replace('_taper_','_blank_')))
  V=shape(t,ny,nz); out={}
  for label,state in states:
   target=ref@state
   cc=np.linalg.lstsq(V,target,rcond=1e-12)[0];d=V@cc-target
   err=float(np.linalg.norm(d)/np.linalg.norm(target));Q,_=qr(cc,mode='economic')
   out[label]=dict(relative_shape_projection_error=err,minimum_power=float(np.linalg.eigvalsh(Q.conj().T@g@Q).min()),maximum_power=float(np.linalg.eigvalsh(Q.conj().T@g@Q).max()),blank_minimum_power=float(np.linalg.eigvalsh(Q.conj().T@qb@Q).min()))
  print(json.dumps(dict(case=p.name,frozen_on='H85 and H60, identical SAFE incident coordinates',result=out)),flush=True)
