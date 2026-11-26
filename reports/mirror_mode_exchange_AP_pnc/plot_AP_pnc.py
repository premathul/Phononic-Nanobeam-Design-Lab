"""Plot the fixed-frequency mirror diagnostic. Athul Prem | _AP_pnc."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parent
repo=Path(sys.argv[1])
src=repo/'reports/mirror_length_AP_pnc'
rows=[]
for n in range(1,9):
    r=json.loads((src/f'finite_{n}_local_AP_pnc.json').read_text())['result']
    u,v=r['native_incident_root_ids'],r['native_reflection_root_ids']
    a=r['reflection_amplitudes']
    power=(np.asarray(a['real'])**2+np.asarray(a['imag'])**2)/np.asarray(r['incident_powers_w'])
    rows.append([n,*(100*power[v.index(i),u.index(j)] for i,j in [(477,492),(493,476),(493,492),(477,476)])])
with (p/'reflection_AP_pnc.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['periods','492_to_477_percent','476_to_493_percent','492_to_493_percent','476_to_477_percent']);w.writerows(rows)
mpl.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.hashsalt':'modepair_AP_pnc'})
fig,(a,b)=plt.subplots(1,2,figsize=(7.4,3.2),layout='constrained',gridspec_kw={'width_ratios':[1,1.23]})
n,c,d,e,f=np.array(rows).T
a.plot(n,c,'o-',color='#106284',label='3 → 5 and 5 → 3',ms=4,lw=1.35)
a.plot(n,e,'s--',color='#a34527',label='3 → 3',ms=3.5,lw=1.0)
a.plot(n,f,'^:',color='#4b4b4b',label='5 → 5',ms=3.5,lw=1.0)
a.set(xlabel='Mirror periods',ylabel='Reflection (% of incident power)',title='(a) Finite mirror at 1 GHz',xticks=n,ylim=(-1,42))
a.legend(loc='upper left',frameon=False,fontsize=8)
a.grid(axis='y',alpha=.23)
names=['mirror_length_AP_pnc','t100_z2_edge225nm_AP_pnc','t100_z2_edge168p75nm_AP_pnc','t100_z2_edge126p5625nm_AP_pnc','t100_z3_edge126p5625nm_AP_pnc']
tags=['225 / 2 (Mac)','225 / 2','168.75 / 2','126.56 / 2','126.56 / 3']
for y,stem in enumerate(names):
    data=json.loads((p/'results'/f'{stem}.json').read_text())
    r0=[x for x in data['frozen_condensed_roots'] if abs(x['modulus']-1)<1e-6]
    r1=[x for x in data['frozen_condensed_roots'] if x['lead_pair_trace_mass_overlap']>.95]
    assert len(r0)==6 and len(r1)==2
    assert max(x['lead_pair_trace_mass_overlap'] for x in r0)<3e-5
    assert min(x['lead_pair_trace_mass_overlap'] for x in r1)>.998
    assert max(x['residual'] for x in r0+r1)<1e-10
    b.scatter(r0[0]['modulus'],y,facecolors='none',edgecolors='#484848',marker='s',s=27,label='Other symmetry: propagating' if y==0 else None,zorder=3)
    b.scatter(r1[0]['modulus'],y,c='#106284',marker='o',s=27,label='Same symmetry: decaying pair' if y==0 else None,zorder=3)
b.set(xlim=(.91,1.007),ylim=(4.45,-.45),xlabel='Bloch amplitude multiplier |λ| per period',yticks=range(5),yticklabels=tags,title='(b) Archived cell discretizations')
b.set_ylabel('Edge target (nm) / thickness layers',fontsize=8)
b.legend(loc='upper center',bbox_to_anchor=(.5,-.23),frameon=False,fontsize=7.4)
b.grid(axis='x',alpha=.23)
for ext in ['pdf','svg','png']:
    fig.savefig(p/f'mirror_mode_exchange_AP_pnc.{ext}',dpi=260)
