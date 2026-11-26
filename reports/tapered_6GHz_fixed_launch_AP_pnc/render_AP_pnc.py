"""Athul Prem | _AP_pnc. Figure and CSV from the checked receipt, without a FEM rerun."""
import csv, json
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root = Path(__file__).resolve().parent
rows = json.loads((root/'fixed_launch_receipt_AP_pnc.json').read_text())
with (root/'fixed_launch_AP_pnc.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['case','h_nm','port','retarded_lead_dx_nm','launch','beam','T_fraction','R_fraction','trace_L2_relative_residual'])
    for name,row in rows.items():
        for launch,q in row['launches'].items():
            for beam in ('blank','taper'):
                w.writerow([name,row['H_nm'],row['port'],row['lead_axial_nm'],launch,beam,q[beam]['T'],q[beam]['R'],q['projection_error']])
mpl.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False})
fig,ax=plt.subplots(1,3,figsize=(9,3.25),layout='constrained')
colors=['#004a76','#d05422']
for port,(marker,fill) in {'SAFE':('o',True),'retarded':('s',False)}.items():
    for k,row in rows.items():
        if row['port']!=port: continue
        dx=row['lead_axial_nm']
        x=row['H_nm']+(0 if fill else (2.3 if dx>30 else (-2.3 if dx<15 else .9)))
        for i,label in enumerate(('rejected','passed')):
            a=ax[i]; col=colors[i]
            a.plot(x,100*row['launches'][label]['taper']['T'],marker,ms=5,mfc=col if fill else 'none',mec=col,zorder=5)
        ax[2].plot(x,100*row['blank_worst_coherent_R'],marker,ms=5,mfc='0.25' if fill else 'none',mec='0.25',zorder=5)
labels=['Taper: blocked launch','Taper: passing launch','Straight control: reflection']
ylabels=['Power transmitted (%)','Power transmitted (%)','Power reflected (%)']
for i,a in enumerate(ax):
    a.set_xticks([85,60,43,33]); a.set_xlim(92,26)
    a.set_xlabel('Interior mesh target h (nm)')
    a.set_ylabel(ylabels[i]); a.set_title(labels[i],fontsize=9,loc='left',pad=8)
    a.grid(axis='y',alpha=.20)
    a.axvspan(51,26,color='0.94',zorder=0)
    a.text(.96,.08 if i==1 else .91,'Holdout',transform=a.transAxes,ha='right',fontsize=8,color='0.28')
    if i!=1:a.set_yscale('log')
ax[0].set_ylim(1.4e-5,2e-3)
ax[0].axhline(.001,color='0.45',ls='--',lw=.8)
ax[1].set_ylim(99.1,100.1); ax[1].axhline(99.4,color='0.45',ls='--',lw=.8)
ax[2].set_ylim(5e-4,.17)
ax[2].legend(handles=[Line2D([],[],marker='o',ls='',color='0.25',label='SAFE'),Line2D([],[],marker='s',ls='',mfc='none',color='0.25',label='Retarded discrete')],loc='upper right',bbox_to_anchor=(1,.79),frameon=False,fontsize=7)
fig.suptitle('6 GHz | 26-hole taper | lossless bulk-Ge model',fontsize=11)
fig.text(.995,.004,"Athul Prem | athul_simulation_pnc_6GHz",ha="right",va="bottom",fontsize=6,color="0.35")
for suffix in ['svg','png','pdf']:
    fig.savefig(root/f'two_launches_AP_pnc.{suffix}',dpi=300)
