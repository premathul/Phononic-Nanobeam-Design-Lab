"""Independent matched-port 6 GHz fixed-launch comparison. Athul Prem | _AP_pnc."""
import hashlib,json,sys
from pathlib import Path
import numpy as np

def check(archive,pin):
 b=archive.read_bytes(); assert hashlib.sha256(b).hexdigest()==pin
 return np.load(archive,allow_pickle=False)
dirs=sorted(Path(sys.argv[1]).glob('edge*_6GHz*AP_pnc'))
raw=[]
for p in dirs:
    if not (p/'map_AP_pnc.json').is_file(): continue
    a=json.loads((p/'map_AP_pnc.json').read_text());t=a['points'][0]
    if t['status']!='accepted':continue
    ref=t['accepted']['summary'];s=p/ref['path']; assert hashlib.sha256(s.read_bytes()).hexdigest()==ref['sha256']
    d=json.loads(s.read_text()); assert t['qualification']['coherent_power']['status']=='passed'
    data=check(s.parent/'selected_fields_AP_pnc.npz',d['output_sha256']['selected_fields_AP_pnc.npz'])
    meta=a['shared']['artifacts']['port_section']
    section=check(p/meta['path'],meta['sha256'])
    U=data['port_displacements'][:,data['incident_mode_indices']]
    raw.append((p.name,section,U,data['transmission_amplitudes'],data['reflection_amplitudes'],
                float(d['configuration']['mesh']['central_max_edge_nm'])))
raw.sort(key=lambda r:-r[-1])
name,s,U,_,_,_=raw[0]; M=s['M']
port=[]
for name,sc,V,T,R,edge in raw:
    assert all(np.array_equal(s[k],sc[k]) for k in ('points_m','triangles','M','K0','Q','G'))
    z=np.diag(U.conj().T@M@V)
    o=abs(z)/np.sqrt(np.diag(U.conj().T@M@U).real*np.diag(V.conj().T@M@V).real)
    assert min(o)>1-1e-10
    phase=z.conj()/abs(z)
    G=(T*phase).conj().T@(T*phase);H=(R*phase).conj().T@(R*phase)
    assert np.linalg.norm(G+H-np.eye(len(o)),ord=2)<1e-8
    e,V=np.linalg.eigh(G);port.append((edge,e,V,G))
dim=3
cross=[]
for anchor,e,V,G in port:
    Q=V[:,:dim]
    for mesh,_,W,H in port:
        cross.append(dict(anchor_nm=anchor,other_nm=mesh,attenuated_subspace_dimension=dim,
             power_pct_bounds=(100*np.linalg.eigvalsh(Q.conj().T@H@Q)).tolist(),
             canonical_correlations=np.linalg.svd(Q.conj().T@W[:,:dim],compute_uv=False).tolist()))
combined=np.mean([r[-1] for r in port],axis=0)
_,V=np.linalg.eigh(combined);Q=V[:,:dim]
common=[dict(edge_nm=a,bounds_pct=(100*np.linalg.eigvalsh(Q.conj().T@G@Q)).tolist()) for a,_,_,G in port]
_,V_train=np.linalg.eigh(np.mean([r[-1] for r in port[:2]],axis=0))
Q_train=V_train[:,:2]
held=[dict(edge_nm=a,bounds_pct=(100*np.linalg.eigvalsh(Q_train.conj().T@G@Q_train)).tolist())
      for a,_,_,G in port]
d=dict(two_dimensional_coarse_mid_trained_span=held,
       two_dimensional_training='Two lowest eigenvectors of mean Gram on coarse and intermediate meshes only; finest excluded from training.',
       common_span_design='Three least-transmitting eigendirections of equal-weight mean of sampled Grams; trained on all meshes, not out-of-sample.',
       cross_checks=cross,common_span=common,edge_nm=[r[0] for r in port],
       power_increasing_spectrum_pct=[(100*r[1]).tolist() for r in port])
Path(sys.argv[2]).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
for r in cross:print(r)
print('COMMON',common); print('HELD',held)
