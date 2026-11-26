"""Athul Prem | _AP_pnc. Re-evaluate two pre-frozen launches on paired blank/taper meshes."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
from fixed_launch_AP_pnc import data, ref, states, shape, root

cases = {}
for h, y, z, rim in ((85,16,2,32),(60,16,2,32),(43,20,3,40),(33,20,3,40)):
    stem = f'section_H{h}_ny{y}_nz{z}_rim{rim}'
    for n in (None, 384, 768, 1536):
        prefix = f'retarded_dxN{n}_' if n else ''
        tp = root / f'{prefix}{stem}_taper_AP_pnc.npz'
        bp = root / f'{prefix}{stem}_blank_AP_pnc.npz'
        if not (tp.exists() and bp.exists()):
            continue
        a,b = data(tp),data(bp)
        if n:
            base = data(root / f'{stem}_taper_AP_pnc.npz')
            assert np.array_equal(a['incident_u'],b['incident_u'])
            shapes = {**base,'up':a['incident_u']}
        else:
            assert np.array_equal(a['up'],b['up'])
            shapes = a
        v = shape(shapes,y,z)
        launches = {}
        for label, state in states:
            goal = ref @ state[:,0]
            c = np.linalg.lstsq(v,goal,rcond=1e-12)[0]
            err = np.linalg.norm(v@c-goal)/np.linalg.norm(goal)
            c /= np.linalg.norm(c)
            m = len(c)
            def pair(d):
                out = d['S'][:, :m] @ c
                return dict(R=float(np.linalg.norm(out[:m])**2),T=float(np.linalg.norm(out[m:])**2))
            launches[label] = dict(projection_error=float(err),blank=pair(b),taper=pair(a))
        key = f'H{h}_'+(f'D{n}' if n else 'SAFE')
        row=dict(H_nm=h,port='SAFE' if n is None else 'retarded',lead_axial_nm=None if n is None else 16650/n,launches=launches,
          blank_worst_coherent_R=float(np.linalg.norm(b['S'][:m,:m],2)**2),
          unitary_residual=float(np.linalg.norm(a['S'].conj().T@a['S']-np.eye(2*m),2)),
          hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (bp,tp)})
        cases[key]=row
print(json.dumps(cases,indent=2))
