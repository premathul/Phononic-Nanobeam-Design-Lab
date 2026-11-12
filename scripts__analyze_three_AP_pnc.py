"""Create three-structure suppression tables and probe-response figures."""
from pathlib import Path
import csv,math
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; T=ROOT/'results/tables'; F=ROOT/'results/figures'
def regions(rows,key,thr):
 out=[];i=0
 while i<len(rows):
  if float(rows[i][key])>thr:i+=1;continue
  j=i
  while j+1<len(rows) and float(rows[j+1][key])<=thr:j+=1
  s=rows[i:j+1];out.append({'metric':key,'threshold_dB':thr,'lower_frequency_GHz':s[0]['frequency_GHz'],'upper_frequency_GHz':s[-1]['frequency_GHz'],'bandwidth_GHz':float(s[-1]['frequency_GHz'])-float(s[0]['frequency_GHz']),'minimum_dB':min(float(x[key]) for x in s)});i=j+1
 return out
rows=list(csv.DictReader(open(T/'three_structure_frequency_sweep.csv')));out=[]
for key in ['periodic_relative_dB','defect_relative_dB']:
 for th in (-10.,-20.,-30.):out+=regions(rows,key,th)
with open(T/'three_structure_suppression_regions.csv','w',newline='') as f:
 keys=['metric','threshold_dB','lower_frequency_GHz','upper_frequency_GHz','bandwidth_GHz','minimum_dB'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(out)
fig,ax=plt.subplots(figsize=(8,4));
for key,label in [('periodic_relative_dB','periodic mirror'),('defect_relative_dB','tapered defect')]:ax.plot([float(r['frequency_GHz']) for r in rows],[float(r[key]) for r in rows],label=label)
ax.set(xlabel='frequency (GHz)',ylabel='relative transmission (dB)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(F/'three_structure_transmission.png',dpi=220);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));
for key,label in [('unpatterned_out','unpatterned'),('periodic_out','periodic mirror'),('tapered_defect_out','tapered defect')]:ax.plot([float(r['frequency_GHz']) for r in rows],[float(r[key]) for r in rows],label=label)
ax.set(xlabel='frequency (GHz)',ylabel='P3 mean |u| (m)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(F/'probe_response.png',dpi=220);plt.close(fig)
print('THREE_ANALYSIS_PASS',len(out),'suppression rows')
