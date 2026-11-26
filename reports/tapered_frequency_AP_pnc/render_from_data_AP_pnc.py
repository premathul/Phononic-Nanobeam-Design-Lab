"""Render both tapered-device figures from compact pinned data. Athul Prem | _AP_pnc."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
out=Path(sys.argv[1])
records=json.loads((out/'summary_AP_pnc.json').read_text())
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
a.plot([],[],'o',mfc='none',mec='#af4b24',ms=3,label='Individual lead modes (offset by 25 MHz)')
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

with np.load(out/'midplane_vertices_AP_pnc.npz',allow_pickle=False) as data:
    xy,faces,amp_nm=[data[x] for x in ('xy_um','tris','displacement_norm_nm_per_sqrt_W')]
f,ax=plt.subplots(figsize=(8,1.95),layout='constrained')
tri=Triangulation(*xy.T,triangles=faces)
im=ax.tripcolor(tri,amp_nm,shading='gouraud',cmap='viridis')
ax.set_aspect('equal')
ax.set(xlabel='Position along beam (µm)',ylabel='Width (µm)',title='6 GHz midplane displacement |u|, native incident mode 1')
f.colorbar(im,ax=ax,label='nm/√W',shrink=.73)
f.savefig(out/'tapered_6GHz_field_AP_pnc.png',dpi=260)
