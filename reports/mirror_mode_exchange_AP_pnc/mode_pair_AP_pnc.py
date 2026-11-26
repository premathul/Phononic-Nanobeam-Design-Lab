from tools import run_mirror_cell_phase_AP_pnc as cell, finite_mirror_response_AP_pnc as f
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
import numpy as np,json
from pathlib import Path
import argparse, hashlib, platform, scipy, gmsh, skfem
from scipy.linalg import eig
from tools.bloch_factors_AP_pnc import _companion
p=argparse.ArgumentParser(description="Test native and Bloch mode symmetry from a frozen cell archive. Athul Prem | _AP_pnc.")
p.add_argument("archive",type=Path)
p.add_argument("output",type=Path)
args=p.parse_args()
folder=args.archive
receipt=json.loads((folder/'summary_AP_pnc.json').read_text())
assert receipt['status']=='accepted'
assert hashlib.sha256((folder/'traces_AP_pnc.npz').read_bytes()).hexdigest()==receipt['trace_arrays']['sha256']
target=receipt['inputs']['mesh']['central_target_nm']
layers=receipt['inputs']['mesh']['thickness_layers']
spec=cell._case_spec(100,target,layers)
sys=cell.assemble_case(100,target,layers)
port=assemble_port(sys,spec.material,max_dofs=spec.inputs['mesh']['max_face_dofs'])
m=solve_port_modes(port,1e9)
idxs=[s[m.propagating[s]] for s in [m.positive_indices,m.negative_indices]]
B=port.basis
a=np.load(folder/'traces_AP_pnc.npz')
ix=f._native_trace_order(a,port)
masserr=float(np.max(abs(a['trace_mass'][ix]-port.M))/np.max(abs(port.M)))
print('matched trace mass max rel',masserr,'shape',port.M.shape,flush=True)
print('lead indices',[(x.tolist(),m.wavenumbers_rad_m[x].real.tolist()) for x in idxs],flush=True)
out = {'positive':[(int(i),float(m.wavenumbers_rad_m[i].real)) for i in idxs[0]],'negative':[(int(i),float(m.wavenumbers_rad_m[i].real)) for i in idxs[1]],'trace_mass_max_rel':masserr}
P_by_reflection=[]
for c in (0,1):
    coords=B.doflocs.copy()
    coords[c]=-coords[c] if c==0 else coords[c].min()+coords[c].max()-coords[c]
    coords=.5*(B.doflocs.min(axis=1)+B.doflocs.max(axis=1))[:,None]+(1-3e-11)*(coords-.5*(B.doflocs.min(axis=1)+B.doflocs.max(axis=1))[:,None])
    P=np.zeros((B.N,B.N))
    for n, ((_,bs),inds) in enumerate(zip(B.split(np.zeros(B.N)),B.split_indices())):
        P[np.ix_(inds,inds)]=(-1 if n==c+1 else 1)*bs.probes(coords[:,inds]).toarray()
    P_by_reflection.append(P)
    def parity(u):
        norm=np.vdot(u,port.M@u).real
        p=np.vdot(u,port.M@(P@u))/norm
        e=min(np.sqrt(np.vdot(P@u-j*u,port.M@(P@u-j*u)).real/norm) for j in [-1,1])
        return [p.real,float(e)]
    out['reflect_y' if c==0 else 'reflect_z']={
       'forward':[parity(m.displacements[:,i]) for i in idxs[0]],
       'backward':[parity(m.displacements[:,i]) for i in idxs[1]],
       'bloch':[parity(u) for u in a['traces'][ix[0].ravel(),:].T]}
print(json.dumps(out),flush=True)

n=port.M.shape[0]
ll,lr,rl,rr=[a[key][ix] for key in ('LL','LR','RL','RR')]
A,C=_companion(np.block([[ll,lr],[rl,rr]]),n)
e,V=eig(A,C,homogeneous_eigvals=True)
positive_pair=m.displacements[:,idxs[0][[2,4]]]
negative_pair=m.displacements[:,idxs[1][[2,4]]]
pair=np.column_stack((positive_pair,negative_pair))
mass=port.M
inv=np.linalg.pinv(pair.conj().T@mass@pair,rcond=1e-10)
a0=float(a['period_m'])
results=[]
for i,(alpha,beta) in enumerate(e.T):
    if not beta or not alpha: continue
    lam=complex(alpha/beta)
    if not .4 <= abs(lam) <= 1.000001: continue
    u=V[:n,i]
    if not np.any(u): continue
    lhs=(rl+(ll+rr)*lam+lr*lam**2)@u
    scale=np.linalg.norm(rl@u)+abs(lam)*np.linalg.norm((ll+rr)@u)+abs(lam)**2*np.linalg.norm(lr@u)
    norm=np.vdot(u,mass@u).real
    s=[]
    for P in P_by_reflection:
        pu=P@u; p=np.vdot(u,mass@pu).real/norm
        diff=pu-(1 if p>=0 else -1)*u
        s.append([float(p),float(np.sqrt(np.vdot(diff,mass@diff).real/norm))])
    coeff=pair.conj().T@mass@u
    results.append(dict(modulus=abs(lam),phase_rad=float(np.angle(lam)),atten_per_period=float(-np.log(abs(lam))),residual=float(np.linalg.norm(lhs)/scale),symmetry=s,lead_pair_trace_mass_overlap=float(np.vdot(coeff,inv@coeff).real/norm)))
results.sort(key=lambda r:(-r['modulus'],r['phase_rad']))
for r in results: print(r,flush=True)
out.update({'archive':str(folder),'archive_sha256':receipt['trace_arrays']['sha256'],
            'port_operator_sha256':f._array_digest(port.K0,port.G,port.Q,port.M),
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'gmsh':gmsh.__version__,'scikit-fem':skfem.__version__},
            'mesh':[target,layers],'frozen_condensed_roots':results,
            'homogeneous_total':e.shape[1],'unit_root_count':sum(abs(x['modulus']-1)<1e-6 for x in results)})
with args.output.open('x') as f_out:
    f_out.write(json.dumps(out,indent=2,allow_nan=False)+'\n')
