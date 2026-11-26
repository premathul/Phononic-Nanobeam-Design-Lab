"""Build the 26-hole transmission-eigenchannel and 6 GHz field figures. Athul Prem | _AP_pnc."""
import argparse,csv,hashlib,json
from pathlib import Path
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('campaign',type=Path);p.add_argument('output',type=Path)
args=p.parse_args()
out=args.output;out.mkdir(exist_ok=True)
records=[]
tables=[]
field_donor=None
files=sorted(args.campaign.glob('edge*/map_AP_pnc.json'))
for path in files:
    a=json.loads(path.read_text())
    label=path.parent.name.split('_')[0].removeprefix('edge').replace('p','.')
    with (path.parent/'scattering_map_AP_pnc.csv').open(newline='') as f:
        table=list(csv.DictReader(f))
    for point in a['points']:
        if point['status']!='accepted': continue
        freq=point['frequency_GHz']
        n=len(point['incoming_channels'])
        M={kind:np.full((n,n),np.nan+1j*np.nan) for kind in ('transmission','reflection')}
        for r in table:
            if float(r['frequency_GHz'])!=freq:continue
            i,j=int(r['outgoing_channel'])-1,int(r['incident_channel'])-1
            z=complex(float(r['amplitude_real']),float(r['amplitude_imag']))
            assert np.isnan(M[r['kind']][i,j])
            M[r['kind']][i,j]=z
            tables.append(dict(central_edge_target_nm=label,**r))
        T,R=M['transmission'],M['reflection']
        assert np.isfinite(T).all() and np.isfinite(R).all()
        G=T.conj().T@T; H=R.conj().T@R
        defect=float(np.linalg.norm(G+H-np.eye(n),2))
        assert defect<1e-8 and point['qualification']['coherent_power']['status']=='passed'
        assert np.max(abs(G.diagonal().real-[x['transmission'] for x in point['incoming_channels']]))<1e-10
        s=point['accepted']['summary'];sp=path.parent/s['path'];data=sp.read_bytes()
        assert s['sha256']==hashlib.sha256(data).hexdigest()
        d=json.loads(data)
        assert d['hole_count']==26 and d['power_balance_passed'] is True
        native_residual=d['maximum_relative_solver_residual']
        assert native_residual<1e-9
        tr=np.sort(np.linalg.svd(T,compute_uv=False)**2)
        rec=dict(frequency_GHz=freq,hole_count=d['hole_count'],central_edge_target_nm=float(label),
                 thickness_nm=d['configuration']['geometry']['thickness_nm'],body_dofs=d['ndof'],channels=n,
                 coherent_transmission_pct=(100*tr).tolist(),individual_transmission_pct=(100*G.diagonal().real).tolist(),
                 reflection_matrix_pct=(100*abs(R)**2).tolist(),max_relative_residual=native_residual,
                 flux_power_imbalance=d['maximum_flux_power_imbalance'],coherent_power_defect=defect,
                 peak_solver_mib=point['resources']['scattering']['peak_process_memory_mib'],
                 mesh_sha256=a['shared']['artifacts']['mesh']['sha256'],source_git_commit=a['source_git_commit'],
                 receipt_sha256=s['sha256'],coefficient_map_sha256=hashlib.sha256((path.parent/'scattering_map_AP_pnc.csv').read_bytes()).hexdigest())
        records.append(rec)
        if freq==6 and float(label)==225:
            field_donor=(sp.parent/'selected_fields_AP_pnc.npz',d)
records.sort(key=lambda x:(x['frequency_GHz'],-x['central_edge_target_nm']))
assert records
(out/'summary_AP_pnc.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
with (out/'native_coefficients_AP_pnc.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)

mpl.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.hashsalt':'taper_frequency_AP_pnc'})
fig,(a,b)=plt.subplots(1,2,figsize=(7.55,3.35),layout='constrained')
base=[r for r in records if r['central_edge_target_nm']==225]
for r in base:
    freq=r['frequency_GHz'];eig=r['coherent_transmission_pct']; ip=r['individual_transmission_pct']
    a.vlines(freq,eig[0],eig[-1],colors='#174f72',lw=2)
    a.scatter([freq]*len(eig),eig,s=9,c='#174f72',zorder=3)
    a.scatter([freq+.025]*len(ip),ip,s=6,facecolors='none',edgecolors='#af4b24',alpha=.55,zorder=2)
a.set(xlabel='Frequency (GHz)',ylabel='Transmitted power (%)',title='(a) 26-hole tapered device: sampled frequencies',ylim=(-3,103))
a.grid(axis='y',alpha=.25)
a.plot([],[],'o',c='#174f72',ms=3,label='Coherent eigenchannels')
a.plot([],[],'o',mfc='none',mec='#af4b24',ms=3,label='Individual lead modes (+25 MHz offset)')
a.legend(loc='lower left',frameon=False,fontsize=7.2)
six=[r for r in records if r['frequency_GHz']==6]
for i,r in enumerate(six):
    ip=r['individual_transmission_pct'];e=r['coherent_transmission_pct']
    b.vlines(i,e[0],e[-1],colors='#174f72',lw=2)
    b.scatter([i]*len(e),e,s=9,c='#174f72',zorder=3)
    b.scatter([i+.045]*len(ip),ip,s=6,facecolors='none',edgecolors='#af4b24',alpha=.55,zorder=2)
b.set(xticks=range(len(six)),xticklabels=[f"{r['central_edge_target_nm']:.2f}\n{r['channels']} channels" for r in six],
      xlabel='Central edge target (nm)',title='(b) 6 GHz mesh comparison',ylim=(-3,103))
b.grid(axis='y',alpha=.25)
for ext in ('svg','pdf','png'):fig.savefig(out/f'tapered_frequency_AP_pnc.{ext}',dpi=230)

if field_donor:
    src,d=field_donor
    assert hashlib.sha256(src.read_bytes()).hexdigest()==d['output_sha256'][src.name]
    with np.load(src,allow_pickle=False) as w:
        xyz,tets=w['points_m'],w['tetrahedra']
        fcs=np.sort(np.concatenate([tets[:,c] for c in itertools.combinations(range(4),3)]),axis=1)
        inside=np.all(abs(xyz[fcs,2])<1e-14,axis=1)
        plane=np.unique(fcs[inside],axis=0)
        mask=np.unique(plane)
        index=np.full(len(xyz),-1);index[mask]=np.arange(len(mask))
        faces=index[plane]
        u=w['displacements_per_sqrt_w'][w['nodal_dofs'][:,mask],0]
        amp_nm=np.linalg.norm(u,axis=0)*1e9
        xy=xyz[mask,:2]*1e6
        root=int(w['incident_mode_indices'][0])
    assert len(faces)>100 and np.isfinite(amp_nm).all()
    np.savez_compressed(out/'midplane_vertices_AP_pnc.npz',xy_um=xy,tris=faces,
                       displacement_norm_nm_per_sqrt_W=amp_nm,incident_root=root,
                       field_source_sha256=d['output_sha256'][src.name])
    f,ax=plt.subplots(figsize=(8,1.95),layout='constrained')
    tri=Triangulation(*xy.T,triangles=faces)
    im=ax.tripcolor(tri,amp_nm,shading='gouraud',cmap='viridis')
    ax.set_aspect('equal')
    ax.set(xlabel='Position along beam (µm)',ylabel='Width (µm)',title='6 GHz midplane displacement |u|, native incident mode 1')
    f.colorbar(im,ax=ax,label='nm/√W',shrink=.73)
    f.savefig(out/'tapered_6GHz_field_AP_pnc.png',dpi=260)
