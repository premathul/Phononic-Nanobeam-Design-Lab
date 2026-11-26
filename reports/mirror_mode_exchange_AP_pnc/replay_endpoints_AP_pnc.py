from tools import run_mirror_cell_phase_AP_pnc as cell, finite_mirror_response_AP_pnc as f
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
import numpy as np,json
from pathlib import Path
folder=Path('reports/mirror_length_AP_pnc')
spec=cell._case_spec(100,225.,2)
sys=cell.assemble_case(100,225.,2)
port=assemble_port(sys,spec.material,max_dofs=spec.inputs['mesh']['max_face_dofs'])
m=solve_port_modes(port,1e9)
idxs=[s[m.propagating[s]] for s in [m.positive_indices,m.negative_indices]]
B=port.basis
a=np.load(folder/'traces_AP_pnc.npz')
ix=f._native_trace_order(a,port)
masserr=float(np.max(abs(a['trace_mass'][ix]-port.M))/np.max(abs(port.M)))
print('matched trace mass max rel',masserr,'shape',port.M.shape,flush=True)
print('lead indices',[(x.tolist(),m.wavenumbers_rad_m[x].real.tolist()) for x in idxs],flush=True)
for n in (1,8):
 r=f.solve_finite_mirror_response(folder/'traces_AP_pnc.npz',m,n)
 old=json.loads((folder/f'finite_{n}_local_AP_pnc.json').read_text())['result']
 def p(x):return np.array(x['real'])**2+np.array(x['imag'])**2
 print(n,r['status'],{key:float(np.max(abs(p(r[key])-p(old[key])))) for key in ('reflection_amplitudes','transmission_amplitudes')},flush=True)
 Path(f'../replay_{n}_AP_pnc.json').write_text(json.dumps(r,indent=2)+'\n')
out = {'positive':[(int(i),float(m.wavenumbers_rad_m[i].real)) for i in idxs[0]],'negative':[(int(i),float(m.wavenumbers_rad_m[i].real)) for i in idxs[1]],'trace_mass_max_rel':masserr}
for c in (0,1):
    coords=B.doflocs.copy()
    coords[c]=-coords[c] if c==0 else coords[c].min()+coords[c].max()-coords[c]
    coords=.5*(B.doflocs.min(axis=1)+B.doflocs.max(axis=1))[:,None]+(1-3e-11)*(coords-.5*(B.doflocs.min(axis=1)+B.doflocs.max(axis=1))[:,None])
    P=np.zeros((B.N,B.N))
    for n, ((_,bs),inds) in enumerate(zip(B.split(np.zeros(B.N)),B.split_indices())):
        P[np.ix_(inds,inds)]=(-1 if n==c+1 else 1)*bs.probes(coords[:,inds]).toarray()
    def parity(u):
        norm=np.vdot(u,port.M@u).real
        p=np.vdot(u,port.M@(P@u))/norm
        e=min(np.sqrt(np.vdot(P@u-j*u,port.M@(P@u-j*u)).real/norm) for j in [-1,1])
        return [p.real,float(e)]
    out['reflect_y' if c==0 else 'reflect_z']={
       'forward':[parity(m.displacements[:,i]) for i in idxs[0]],
       'backward':[parity(m.displacements[:,i]) for i in idxs[1]],
       'bloch':[parity(u) for u in a['traces'][ix[0].ravel(),:].T]}
Path('../exact_port_AP_pnc.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
