"""Isolated 4 K HPGe-requested study; results remain conditional on material data.

Run with the existing simulation Python. No installers, cloud APIs, or external
writes are used. --output creates a new sibling run; --resume is explicit.
"""
from __future__ import annotations
import argparse,atexit,csv,hashlib,json,os,platform,re,shutil,subprocess,sys,time,traceback,uuid
from pathlib import Path
sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parent
PROJECT=SOURCE.parents[2]
ap=argparse.ArgumentParser(description='Portable staged runner. Numerical functions retain the original implementation.')
ap.add_argument('--resume',action='store_true')
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--stage',choices=['smoke','sweep','bands','cavity','robustness','all'],default='smoke')
ap.add_argument('--excel-python',default=sys.executable,help='Interpreter with openpyxl; defaults to this interpreter')
args=ap.parse_args()
OUT=args.output.resolve()
if OUT == SOURCE or OUT.name in ('portable_template','run_20261011T002256Z'):
    ap.error('Output must be a new named run, never the template or archived baseline')
if args.output:
    if OUT.parent!=SOURCE.parent:raise ValueError('--output must be a new sibling under results/4K')
    if args.resume:
        if not OUT.is_dir():raise ValueError('--resume requires an existing run')
    else:
        OUT.mkdir(exist_ok=False)
    # Exclusive per-run lock prevents simultaneous resumes or solver writes.
    lock=OUT/'.run.lock'; lock_token=str(uuid.uuid4())
    try:
        with lock.open('x') as f:f.write(json.dumps({'pid':os.getpid(),'token':lock_token}))
    except FileExistsError:
        raise SystemExit('Run lock exists. Do not resume while a run is active. If interrupted, verify no process is using the run before removing its .run.lock.')
    def release_lock():
        try:
            if json.loads(lock.read_text()).get('token')==lock_token:lock.unlink()
        except (OSError,ValueError):pass
    atexit.register(release_lock)
    for name in ('config.json','MATERIAL_PROVENANCE.md','NUMERICAL_METHODS.md','run.py','numerics.py','finalize.py','verify_eigen_scaling.py','check_legacy_bloch.py'):
        if not args.resume:shutil.copy2(SOURCE/name,OUT/name)
for name in ('figures','tables','fields','logs','mesh','inputs'):(OUT/name).mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'logs'/'matplotlib-cache')
os.environ['XDG_CACHE_HOME']=str(OUT/'logs'/'cache')
os.environ['MPLBACKEND']='Agg'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import scipy
import gmsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.optimize import linear_sum_assignment
import numerics as n

if args.resume:
    for filename in ('run.py','numerics.py'):
        if (OUT/filename).read_bytes() != (SOURCE/filename).read_bytes():
            raise ValueError('Resume source differs from template; use the original template or a new run')
    source_hash_file=OUT/'logs'/'input_source_hashes.json'
    if not source_hash_file.exists():
        raise ValueError('Cannot safely resume without prior source hashes; choose a new run')
    previous_hashes=json.loads(source_hash_file.read_text())
    for rel in ('scripts/solve_direct_fem.py','config.py','results/convergence_and_localization/fields/cell_h20.msh','results/convergence_and_localization/fields/cell_h12.msh','results/convergence_and_localization/fields/cell_h10.msh'):
        found=[digest for key,digest in previous_hashes.items() if key.replace(chr(92),'/').endswith('/'+rel)]
        current=hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest()
        if len(found)!=1 or found[0]!=current:
            raise ValueError('Shared source changed since prior run: '+rel+'; choose a new run')
    oldcfg=json.loads((OUT/'config.json').read_text()); newcfg=json.loads((SOURCE/'config.json').read_text())
    oldcfg.pop('run_id',None);newcfg.pop('run_id',None)
    if oldcfg != newcfg:raise ValueError('Resume configuration differs from template')
CFG=json.loads((OUT/'config.json').read_text())
if args.output:
    CFG['run_id']=OUT.name
    (OUT/'config.json').write_text(json.dumps(CFG,indent=2)+'\n')
FIG=OUT/'figures';TAB=OUT/'tables';FLD=OUT/'fields';LOG=OUT/'logs';MESH=OUT/'mesh'
TH=CFG['validation_thresholds']


def log(*items):
    msg=time.strftime('%Y-%m-%d %H:%M:%S ')+' '.join(map(str,items));print(msg,flush=True)
    with (LOG/'run.log').open('a') as f:f.write(msg+'\n')


def js(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def csvout(path,rows,keys=None):
    keys=keys or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def csvread(path):return list(csv.DictReader(path.open()))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def geometry():
    wb=OUT/'inputs'/'nanobeam_hole_array.xlsx'
    derived=OUT/'inputs'/'hole_geometry_derived.csv'
    for p in (wb,derived):
        if not p.exists():shutil.copy2(PROJECT/'data'/p.name,p)
        if sha(p)!=sha(PROJECT/'data'/p.name):raise ValueError('Input differs from original workbook/derived data')
    code='import json,sys; from openpyxl import load_workbook; b=load_workbook(sys.argv[1],read_only=True,data_only=True); print(json.dumps({"sheets":b.sheetnames,"rows":list(b["Nanobeam Hole Array"].values)}))'
    raw=json.loads(subprocess.check_output([args.excel_python,'-c',code,str(wb)],text=True))
    headers=[str(x or '').strip() for x in raw['rows'][0]]
    def col(prefix):
        found=[i for i,x in enumerate(headers) if x.startswith(prefix)]
        if len(found)!=1:raise ValueError('Ambiguous workbook column '+prefix)
        return found[0]
    cols={k:col(v) for k,v in {'number':'Hole #','x':'Position (nm)','d':'Diameter (nm)','s':'Center-to-center distance','g':'Edge-to-edge gap','region':'Region'}.items()}
    def num(v):
        if isinstance(v,(int,float)):return float(v)
        match=re.fullmatch(r'\s*([-+]?\d+(?:\.\d+)?)\s*',str(v))
        return float(match.group(1)) if match else None
    rows=[]
    for row in raw['rows'][1:]:
        number=num(row[cols['number']])
        if number is None:continue
        r={'hole_number':int(number),'position_nm':num(row[cols['x']]),'diameter_nm':num(row[cols['d']]),
           'spreadsheet_spacing_nm':num(row[cols['s']]),'spreadsheet_gap_nm':num(row[cols['g']]),'region':str(row[cols['region']]).strip()}
        if r['position_nm'] is None or r['diameter_nm'] is None:raise ValueError('Non-numeric geometry')
        if rows:
            r['calculated_spacing_nm']=r['position_nm']-rows[-1]['position_nm']
            r['calculated_gap_nm']=r['calculated_spacing_nm']-(r['diameter_nm']+rows[-1]['diameter_nm'])/2
            for a,b in [('spreadsheet_spacing_nm','calculated_spacing_nm'),('spreadsheet_gap_nm','calculated_gap_nm')]:
                if r[a] is None or abs(r[a]-r[b])>1e-9:raise ValueError('Workbook spacing/gap discrepancy')
            if r['calculated_gap_nm']<=0:raise ValueError('Overlapping holes')
        if r['diameter_nm']>=CFG['beam_width_nm']:raise ValueError('Hole leaves beam')
        r['status']='PASS';rows.append(r)
    prior=csvread(derived)
    if len(rows)!=26 or len(prior)!=26:raise ValueError('Expected 26 holes')
    for a,b in zip(rows,prior):
        for k in ('hole_number','position_nm','diameter_nm'):
            if abs(a[k]-float(b[k]))>1e-9:raise ValueError('Workbook/derived mismatch')
    holes=[(r['position_nm'],r['diameter_nm']) for r in rows]
    if holes[12][0]!=-250 or holes[13][0]!=250:raise ValueError('Central positions differ')
    csvout(TAB/'hole_validation.csv',rows)
    report={'status':'PASS','sheet_names':raw['sheets'],'exact_columns':headers,'hole_count':len(holes),'workbook_sha256':sha(wb),'derived_sha256':sha(derived)}
    js(TAB/'geometry_validation.json',report)
    bounds=(min(x-d/2 for x,d in holes)-CFG['left_lead_nm'],max(x+d/2 for x,d in holes)+CFG['right_lead_nm'],CFG['beam_width_nm'])
    periodic=[((i-12.5)*550,250.) for i in range(26)]
    variants={'reference':[],'periodic':periodic,'defect':holes}
    js(OUT/'inputs'/'geometry.json',{'bounds_nm':bounds,'structures':variants,
       'periodic_control_note':'Same total beam dimensions. Uniform 550 nm holes extend 175 nm farther at each end than the tapered holes; the common source, probes and sponge lie outside both arrays.'})
    from matplotlib.patches import Circle,Rectangle
    fig,axes=plt.subplots(3,1,figsize=(13,5))
    for ax,(name,hh) in zip(axes,variants.items()):
        ax.add_patch(Rectangle((bounds[0]/1000,-.5),(bounds[1]-bounds[0])/1000,1,facecolor='#d6e3ee',edgecolor='#36495a'))
        for j,(x,d) in enumerate(hh):
            ax.add_patch(Circle((x/1000,0),d/2000,facecolor='white',edgecolor='#a44245' if abs(x)<2600 else '#286080'))
            if name=='defect':ax.text(x/1000,.23,str(j+1),ha='center',fontsize=5)
        ax.set(xlim=(bounds[0]/1000-.1,bounds[1]/1000+.1),ylim=(-.65,.65),aspect='equal',ylabel='y (µm)',title=name)
    axes[-1].set_xlabel('x (µm)');fig.suptitle('Original 26-hole geometry · fixed dimensions · conditional cubic-Ge study')
    fig.tight_layout();fig.savefig(FIG/'geometry.png',dpi=220);plt.close(fig)
    return variants,bounds


def snapshot():
    versions={'python':sys.version,'executable':sys.executable,'architecture':platform.machine(),'numpy':np.__version__,'scipy':scipy.__version__,'gmsh':gmsh.__version__,'matplotlib':matplotlib.__version__}
    js(LOG/'environment.json',versions)
    source_dir=LOG/'source_snapshot';source_dir.mkdir(exist_ok=True)
    hashes={}
    for path in (PROJECT/'scripts'/'solve_direct_fem.py',PROJECT/'config.py',SOURCE/'run.py',SOURCE/'numerics.py',SOURCE/'config.json') + tuple(PROJECT/'results'/'convergence_and_localization'/'fields'/f'cell_h{h}.msh' for h in (20,12,10)):
        dest=source_dir/(sha(path)[:12]+'_'+path.name)
        if not dest.exists():shutil.copy2(path,dest)
        hashes[str(path)]=sha(path)
    js(LOG/'input_source_hashes.json',hashes)
    D,V=n.plane_stress(CFG)
    js(LOG/'material_operator.json',{'material_temperature_calibrated':False,'orientation_measured':False,'D_plane_stress_Pa':D.tolist(),'C_rotated_voigt_Pa':V.tolist(),'temperature_K':4,'bath_only':True})
    js(LOG/'probe_definitions.json',{'half_width_nm':80,'centers_nm':{'P0':-7900,'P1':-7150,'P2':0,'P3':7150,'P4':7750},'averaging':'area-weighted seven-point triangle quadrature','source':'1 Pa uniform longitudinal traction at left edge; unit model thickness','sponge':'quadratic mass-proportional numerical damping, zero for |x| <=7300 nm; P3 is upstream of sponge; P4 is within sponge'})


def prepared(name,h=30,bulk=40,angle=0):
    path=MESH/f'{name}_h{h}_b{bulk}.msh'
    if not path.exists():
        log('Meshing',name,h,bulk);n.build_mesh(path,VARIANTS[name],h,bulk,BOUNDS)
    mesh=n.load(path,VARIANTS[name]);D,_=n.plane_stress(CFG,angle)
    s=n.assemble(mesh,D,CFG);checks=n.patch_check(s,CFG)
    js(TAB/f'mesh_{name}_h{h}_b{bulk}.json',mesh['summary'])
    js(TAB/f'patch_{name}_h{h}_b{bulk}_angle{angle}.json',checks)
    log('Mesh and patch PASS',name,h,mesh['summary']['nodes'],mesh['summary']['triangles'])
    return s


def fieldplot(s,u,stem,title):
    _,sed,*_=n.fields(s,u)
    np.savez_compressed(FLD/(stem+'.npz'),points_m=s['p'],triangles=s['t'],displacement_complex=u,strain_energy_density=sed)
    tri=mtri.Triangulation(s['p'][:,0]*1e6,s['p'][:,1]*1e6,s['t'])
    for suffix,data,label in [('displacement',np.linalg.norm(u.reshape(-1,2),axis=1),'|U| (m for 1 Pa traction; eigenmodes arbitrary)'),('strain',sed,'cycle-mean strain energy density (eigenmodes arbitrary)')]:
        fig,ax=plt.subplots(figsize=(13,3));im=ax.tripcolor(tri,data,shading='gouraud' if suffix=='displacement' else 'flat',cmap='magma' if suffix=='displacement' else 'viridis')
        fig.colorbar(im,ax=ax,label=label);ax.set(aspect='equal',xlabel='x (µm)',ylabel='y (µm)',title=title+'\n4 K bath; uncalibrated cubic-Ge proxy')
        fig.tight_layout();fig.savefig(FIG/f'{stem}_{suffix}.png',dpi=220);fig.savefig(FIG/f'{stem}_{suffix}.pdf');plt.close(fig)


def compare_readings(ghz,readings):
    r={'frequency_GHz':float(ghz),'temperature_K':4.,'material_status':'UNCALIBRATED_AT_4K'}
    for name,row in readings.items():
        r.update({name+'_'+k:v for k,v in row.items() if k!='frequency_GHz'})
        r[name+'_within_beam_dB']=20*np.log10(row['P3_u_m']/row['P1_u_m'])
    for name in ('periodic','defect'):
        if name in readings:r[name+'_relative_dB']=float(20*np.log10(readings[name]['P3_u_m']/readings['reference']['P3_u_m']))
    return r


def smoke():
    s=prepared('defect');row,u,sed=n.harmonic(s,4,CFG)
    js(TAB/'single_4GHz.json',row);fieldplot(s,u,'field_4GHz','4 GHz harmonic response')
    log('4 GHz PASS',row['residual'],row['power_balance_relative_error'])
    tri=mtri.Triangulation(s['p'][:,0]*1e6,s['p'][:,1]*1e6,s['t'])
    fig,ax=plt.subplots(figsize=(13,3));ax.triplot(tri,lw=.15);ax.set(aspect='equal',xlabel='x (µm)',ylabel='y (µm)',title='Nominal mesh: '+str(s['summary']['nodes'])+' nodes');fig.tight_layout();fig.savefig(FIG/'mesh.png',dpi=220);plt.close(fig)
    allprep={name:(s if name=='defect' else prepared(name)) for name in VARIANTS}
    rows=[]
    for f in (2,4,6,8,10):
        vals={name:n.harmonic(ss,f,CFG)[0] for name,ss in allprep.items()}
        rows.append(compare_readings(f,vals));csvout(TAB/'five_frequency_comparison.csv',rows)
        log('Five-frequency PASS',f,rows[-1]['defect_relative_dB'])
    # Isotropic assembly regression against the original implementation.
    cell=MESH/'regression_cell_h20.msh'
    if not cell.exists():n.build_mesh(cell,[(0,250)],20,40,(-275,275,1000),True)
    mesh=n.load(cell,[(0,250)])
    ss=n.assemble(mesh,n.legacy.constitutive_plane_stress(),CFG)
    K,M,_=n.legacy.assemble(mesh['p'],mesh['t'])
    check={'K_relative_max_difference':float(abs(K-ss['K']).max()/abs(K).max()),'M_relative_max_difference':float(abs(M-ss['M']).max()/abs(M).max())}
    if max(check.values())>1e-10:raise ValueError('Legacy regression failed')
    D,_=n.plane_stress(CFG);D90,_=n.plane_stress(CFG,90)
    check['cubic_90_degree_invariance']=float(np.max(abs(D-D90))/np.max(abs(D)))
    cubic_iso=dict(CFG,C44_Pa=(CFG['C11_Pa']-CFG['C12_Pa'])/2)
    Di,_=n.plane_stress(cubic_iso);lam=CFG['C12_Pa'];mu=cubic_iso['C44_Pa'];E=mu*(3*lam+2*mu)/(lam+mu);nu=lam/(2*(lam+mu))
    expect=E/(1-nu*nu)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    check['isotropic_limit']=float(np.max(abs(Di-expect))/np.max(abs(expect)))
    if max(check.values())>1e-10:raise ValueError('Constitutive checks failed')
    js(TAB/'independent_operator_checks.json',check)
    return {'four_GHz':'PASS','five_frequencies':'PASS','operator_regression':'PASS'}


def sweep():
    allprep={name:prepared(name) for name in VARIANTS};rows=[]
    fs=np.linspace(1,12,111)
    for f in fs:
        vals={}
        for name,s in allprep.items():
            row,u,sed=n.harmonic(s,float(f),CFG);vals[name]=row
            if name=='defect' and any(abs(f-v)<1e-8 for v in (2,6,8,10)):fieldplot(s,u,f'field_{f:g}GHz',f'{f:g} GHz harmonic response')
        rows.append(compare_readings(f,vals));csvout(TAB/'frequency_sweep.csv',rows)
        log('Sweep',f,'defect dB',rows[-1]['defect_relative_dB'])
    suppressed=[]
    for name in ('periodic','defect'):
        for threshold in (-10,-20,-30):
            indices=np.flatnonzero(np.array([r[name+'_relative_dB'] for r in rows])<=threshold)
            for group in np.split(indices,np.where(np.diff(indices)>1)[0]+1):
                if len(group):suppressed.append({'structure':name,'threshold_dB':threshold,'lower_GHz':float(fs[group[0]]),'upper_GHz':float(fs[group[-1]]),'sample_count':len(group),'minimum_dB':min(rows[i][name+'_relative_dB'] for i in group),'classification':'sampled finite-structure transmission suppression region; provisional'})
    csvout(TAB/'suppression_regions.csv',suppressed,['structure','threshold_dB','lower_GHz','upper_GHz','sample_count','minimum_dB','classification'])
    csvout(TAB/'thermal_4K.csv',[n.thermal(float(f),4) for f in fs])
    fig,ax=plt.subplots(figsize=(10,4.8))
    for name in ('periodic','defect'):ax.plot(fs,[r[name+'_relative_dB'] for r in rows],label=name+' / reference')
    ax.axhline(0,color='gray',ls='--',label='reference');ax.set(xlabel='drive frequency (GHz)',ylabel='P3 displacement ratio (dB)',title='Exploratory transmission · uncalibrated cubic-Ge proxy · 4 K bath');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/'transmission_spectrum.png',dpi=220);fig.savefig(FIG/'transmission_spectrum.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8))
    for probe in ('P1','P2','P3','P4'):ax.semilogy(fs,[r['defect_'+probe+'_u_m'] for r in rows],label=probe)
    ax.set(xlabel='drive frequency (GHz)',ylabel='mean displacement (m), 1 Pa traction',title='Defect probes · conditional material model');ax.legend();fig.tight_layout();fig.savefig(FIG/'probe_response.png',dpi=220);plt.close(fig)
    for label,r in [('strongest_suppression',min(rows,key=lambda r:r['defect_relative_dB'])),('strongest_transmission',max(rows,key=lambda r:r['defect_relative_dB']))]:
        _,u,_=n.harmonic(allprep['defect'],r['frequency_GHz'],CFG);fieldplot(allprep['defect'],u,label,f'{label}: {r["frequency_GHz"]:g} GHz')
    return {'frequencies':len(rows),'linear_solves':3*len(rows),'maximum_residual':max(r[name+'_residual'] for r in rows for name in VARIANTS),'minimum_defect':min(rows,key=lambda r:r['defect_relative_dB'])}


def band_run(h,angle=0):
    path=MESH/f'cell_h{h}.msh'
    if not path.exists():shutil.copy2(PROJECT/'results'/'convergence_and_localization'/'fields'/path.name,path)
    s=n.assemble(n.load(path,[(0,250)]),n.plane_stress(CFG,angle)[0],CFG);n.patch_check(s,CFG)
    values={};resids=[]
    def point(q):
        q=round(float(q),10)
        if q not in values:
            values[q],res=n.bloch_at(s,q,CFG['bloch_branches']);resids.append(res)
            if len(values)%10==0:log('Bloch progress',h,angle,len(values),'k samples')
        return values[q]
    for q in np.linspace(0,1,CFG['bloch_q_points']):point(q)
    # Refine extrema for every sampled candidate gap, twice. Sorted eigenvalues
    # are used only for spectral envelopes, never cavity mode identity.
    for iteration in range(2):
        qs=np.array(sorted(values));b=np.array([values[q] for q in qs]);targets=[]
        for j in range(b.shape[1]-1):
            lo=b[:,j].max();hi=b[:,j+1].min()
            if hi-lo>-.15 and 1<lo<8:
                for i in (int(b[:,j].argmax()),int(b[:,j+1].argmin())):
                    targets.extend(np.linspace(qs[max(0,i-1)],qs[min(len(qs)-1,i+1)],9))
        for q in targets:point(q)
    before={j:(float(np.array(list(values.values()))[:,j].max()),float(np.array(list(values.values()))[:,j+1].min())) for j in range(CFG['bloch_branches']-1)}
    if h==10 and angle==0:
        for q in np.linspace(0,1,81):point(q)
    qs=np.array(sorted(values));b=np.array([values[q] for q in qs]);gaps=[]
    for j in range(b.shape[1]-1):
        lo=float(b[:,j].max());hi=float(b[:,j+1].min())
        if hi>lo and 1<lo<8:gaps.append({'branch_lower':j,'lower_GHz':lo,'upper_GHz':hi,'center_GHz':(lo+hi)/2,'width_GHz':hi-lo,'normalized_width':2*(hi-lo)/(lo+hi)})
    rows=[{'mesh_h_nm':h,'angle_deg':angle,'q_k_over_pi_a':float(q),'branch':j,'frequency_GHz':float(f)} for q,band in zip(qs,b) for j,f in enumerate(band)]
    csvout(TAB/f'bloch_h{h}_angle{angle}.csv',rows)
    sampling=[{'branch_lower':g['branch_lower'],'lower_edge_change_GHz':g['lower_GHz']-before[g['branch_lower']][0],'upper_edge_change_GHz':g['upper_GHz']-before[g['branch_lower']][1]} for g in gaps]
    js(TAB/f'gaps_h{h}_angle{angle}.json',{'gaps':gaps,'mesh':s['summary'],'q_points':len(qs),'maximum_backward_residual':max(resids),'extra_uniform_k_grid_change':sampling})
    log('Bloch',h,angle,'gaps',gaps)
    return gaps,qs,b


def bands():
    summary=[]
    for h in (20,12,10):
        gaps,qs,b=band_run(h)
        summary.extend([dict(g,mesh_h_nm=h) for g in gaps])
    csvout(TAB/'bloch_mesh_convergence.csv',summary,['mesh_h_nm','branch_lower','lower_GHz','upper_GHz','center_GHz','width_GHz','normalized_width'])
    best=json.loads((TAB/'gaps_h10_angle0.json').read_text())['gaps']
    csvout(TAB/'band_gaps_best_estimate.csv',best,['branch_lower','lower_GHz','upper_GHz','center_GHz','width_GHz','normalized_width'])
    fig,ax=plt.subplots(figsize=(8,5))
    for j in range(b.shape[1]):ax.plot(qs,b[:,j],color='#285b8f',lw=.8)
    for g in best:ax.axhspan(g['lower_GHz'],g['upper_GHz'],alpha=.18,color='#d59536')
    ax.set(xlabel='k / (π/a)',ylabel='frequency (GHz)',ylim=(0,8),title='Cubic-Ge [100] plane-stress bands · material not calibrated at 4 K');fig.tight_layout();fig.savefig(FIG/'band_structure.png',dpi=240);fig.savefig(FIG/'band_structure.pdf');plt.close(fig)
    orientation,_,_=band_run(20,45)
    # Independent unperforated long-wavelength extensional speed check.
    path=MESH/'uniform_cell.msh'
    if not path.exists():n.build_mesh(path,[],30,40,(-275,275,1000),True)
    ss=n.assemble(n.load(path,[]),n.plane_stress(CFG)[0],CFG)
    freq,_=n.bloch_at(ss,.002,4)
    D=ss['D'];expected=np.sqrt((D[0,0]-D[0,1]**2/D[1,1])/CFG['rho_kg_m3'])
    k=.002*np.pi/(550e-9);target=expected*k/(2*np.pi*1e9);actual=float(freq[np.argmin(abs(freq-target))]);err=abs(actual/target-1)
    js(TAB/'long_wave_benchmark.json',{'analytic_GHz':target,'FEM_GHz':actual,'relative_error':err,'criterion':0.01,'status':'PASS' if err<.01 else 'FAIL'})
    if err>=.01:raise ValueError('Long-wave benchmark failed')
    return {'gaps':best,'orientation_45deg_gaps':orientation,'long_wave_benchmark':'PASS'}


def cavities():
    gaps=json.loads((TAB/'gaps_h10_angle0.json').read_text())['gaps']
    near=[g for g in gaps if g['lower_GHz']<6]
    target=min(near,key=lambda g:abs(g['center_GHz']-2.8))['center_GHz'] if near else 2.8
    allrows=[];tracked=[];old=None
    for h,bulk in [(30,40),(20,30),(12,20)]:
        s=prepared('defect',h,bulk);rows,U=n.modes(s,target,CFG['eigenmode_count'],CFG,gaps)
        np.savez_compressed(FLD/f'cavity_h{h}.npz',points_m=s['p'],triangles=s['t'],modes=U,frequencies_GHz=[r['frequency_GHz'] for r in rows])
        for r in rows:r.update(mesh_h_nm=h,mesh_bulk_nm=bulk)
        allrows.extend(rows);csvout(TAB/'all_cavity_candidates.csv',allrows)
        ranked=sorted(rows,key=lambda r:(not r['inside_gap'],-r['eta_strain_1um'],-r['eta_strain_2um'],abs(r['x_max_strain_um'])))
        if old is None:
            tracks=[dict(r,track=i+1,overlap_previous=None) for i,r in enumerate(ranked[:4])]
        else:
            # Linear FE interpolation at identical physical sample points, with
            # area weights from the old mesh. Holes/outside points are masked.
            ps,oldU,oldtracks=old
            tri=mtri.Triangulation(s['p'][:,0],s['p'][:,1],s['t'])
            sample=ps['centroid'];a=np.array([oldU[ps['dofs'][:,[0,2,4]],r['mode_index']].mean(axis=1) for r in oldtracks])
            ay=np.array([oldU[ps['dofs'][:,[1,3,5]],r['mode_index']].mean(axis=1) for r in oldtracks])
            mapped=[];valid=np.ones(len(sample),bool)
            for j in range(U.shape[1]):
                ux=mtri.LinearTriInterpolator(tri,U[0::2,j])(sample[:,0],sample[:,1]);uy=mtri.LinearTriInterpolator(tri,U[1::2,j])(sample[:,0],sample[:,1])
                valid &= ~np.ma.getmaskarray(ux)&~np.ma.getmaskarray(uy)
                mapped.append(np.column_stack([ux.filled(0),uy.filled(0)]))
            mapped=np.array(mapped)[:,valid,:];weights=ps['area'][valid]
            prev=np.stack([a[:,valid],ay[:,valid]],axis=2)
            ovs=np.abs(np.einsum('iec,jec,e->ij',prev,mapped,weights))
            norms1=np.sqrt(np.einsum('iec,iec,e->i',prev,prev,weights));norms2=np.sqrt(np.einsum('iec,iec,e->i',mapped,mapped,weights))
            ovs/=norms1[:,None]*norms2[None,:]
            cost=-ovs+.02*np.abs(np.array([r['frequency_GHz'] for r in oldtracks])[:,None]-np.array([r['frequency_GHz'] for r in rows])[None,:])
            ii,jj=linear_sum_assignment(cost);tracks=[]
            for i,j in zip(ii,jj):
                r=dict(rows[j],track=oldtracks[i]['track'],overlap_previous=float(ovs[i,j]),sampled_area_fraction=float(weights.sum()/ps['area'].sum()))
                for key in ('frequency_GHz','eta_strain_1um','eta_strain_2um','eta_strain_2p5um','eta_displacement_1um','eta_displacement_2um','eta_displacement_2p5um'):
                    r['delta_'+key]=r[key]-oldtracks[i][key];r['percent_change_'+key]=100*(r[key]/oldtracks[i][key]-1)
                r['delta_abs_x_max_strain_um']=abs(r['x_max_strain_um'])-abs(oldtracks[i]['x_max_strain_um'])
                tracks.append(r)
        tracked.extend(tracks);csvout(TAB/'cavity_mesh_convergence.csv',tracked)
        old=(s,U,tracks);log('Cavity mesh',h,'top',ranked[0])
    final=sorted(tracks,key=lambda r:(not r['inside_gap'],-r['eta_strain_1um'],-r['eta_strain_2um'],abs(r['x_max_strain_um']),-r['overlap_previous']))
    for i,r in enumerate(final):r['ranking']=i+1
    csvout(TAB/'cavity_ranking.csv',final)
    for i,r in enumerate(final[:4],1):fieldplot(s,U[:,r['mode_index']],f'cavity_rank{i}',f'Candidate {i}: {r["frequency_GHz"]:.6f} GHz')
    for key,title in [('frequency_GHz','frequency (GHz)'),('eta_strain_1um','strain fraction |x|<1 µm'),('eta_displacement_1um','displacement fraction |x|<1 µm')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for track in range(1,5):
            rr=[r for r in tracked if r['track']==track];ax.plot([r['mesh_h_nm'] for r in rr],[r[key] for r in rr],'o-',label=f'track {track}')
        ax.invert_xaxis();ax.set(xlabel='hole/taper target (nm)',ylabel=title);ax.legend();fig.tight_layout();fig.savefig(FIG/f'convergence_{key}.png',dpi=220);plt.close(fig)
    csvout(TAB/'cavity_thermal_4K.csv',[dict(n.thermal(r['frequency_GHz'],4),ranking=r['ranking']) for r in final])
    return {'analyzed_modes':len(allrows),'tracked_modes':len(final),'best':final[0]}


def robustness():
    sweep=csvread(TAB/'frequency_sweep.csv');fmin=float(min(sweep,key=lambda r:float(r['defect_relative_dB']))['frequency_GHz'])
    gaps=json.loads((TAB/'gaps_h10_angle0.json').read_text())['gaps']
    fs=sorted(set([2.8,2.9,3.,4.,5.2,fmin]+[g['center_GHz'] for g in gaps if g['lower_GHz']<6]))
    rows=[]
    for h,bulk in [(30,40),(20,30),(12,20)]:
        ref=prepared('reference',h,bulk);defect=prepared('defect',h,bulk)
        for f in fs:
            rr=n.harmonic(ref,f,CFG)[0];rd=n.harmonic(defect,f,CFG)[0]
            rows.append({'hole_nm':h,'bulk_nm':bulk,'frequency_GHz':f,'relative_dB':float(20*np.log10(rd['P3_u_m']/rr['P3_u_m'])),'max_residual':max(rr['residual'],rd['residual'])})
            csvout(TAB/'transmission_mesh_convergence.csv',rows)
        log('Transmission convergence mesh',h)
    sensitivity=[]
    # Use the same fine mesh for end-loss sensitivity; no intrinsic loss added.
    for loss in CFG['sponge_sensitivity']:
        for f in fs:
            rr=n.harmonic(ref,f,CFG,loss)[0];rd=n.harmonic(defect,f,CFG,loss)[0]
            sensitivity.append({'sponge_loss':loss,'frequency_GHz':f,'relative_dB':float(20*np.log10(rd['P3_u_m']/rr['P3_u_m']))})
            csvout(TAB/'sponge_sensitivity.csv',sensitivity)
    # One direct same-mesh orientation check at 4 GHz.
    rot_ref=n.assemble(ref,n.plane_stress(CFG,45)[0],CFG);rot_def=n.assemble(defect,n.plane_stress(CFG,45)[0],CFG)
    a=n.harmonic(rot_ref,4,CFG)[0];b=n.harmonic(rot_def,4,CFG)[0]
    js(TAB/'orientation_4GHz.json',{'angle_deg':45,'relative_dB':float(20*np.log10(b['P3_u_m']/a['P3_u_m'])),'orientation_status':'ASSUMED_SENSITIVITY'})
    comparisons=[]
    for f in fs:
        vals=[r for r in rows if r['frequency_GHz']==f];lossvals=[r['relative_dB'] for r in sensitivity if r['frequency_GHz']==f]
        change=vals[-1]['relative_dB']-vals[-2]['relative_dB'];span=max(lossvals)-min(lossvals)
        comparisons.append({'frequency_GHz':f,'final_mesh_change_dB':change,'sponge_span_dB':span,'mesh_pass':abs(change)<=TH['transmission_change_dB'],'sponge_pass':span<=TH['transmission_change_dB']})
    csvout(TAB/'transmission_robustness_summary.csv',comparisons)
    return {'all_transmission_mesh_checks_pass':all(r['mesh_pass'] for r in comparisons),'all_sponge_checks_pass':all(r['sponge_pass'] for r in comparisons),'checks':comparisons}


def report():
    sweep=csvread(TAB/'frequency_sweep.csv');best=min(sweep,key=lambda r:float(r['defect_relative_dB']))
    gaps10=json.loads((TAB/'gaps_h10_angle0.json').read_text())['gaps'];gaps12=json.loads((TAB/'gaps_h12_angle0.json').read_text())['gaps']
    changes=[]
    for g in gaps10:
        prev=next((r for r in gaps12 if r['branch_lower']==g['branch_lower']),None)
        if prev:changes.append({'branch_lower':g['branch_lower'],'lower_change_percent':100*(g['lower_GHz']/prev['lower_GHz']-1),'upper_change_percent':100*(g['upper_GHz']/prev['upper_GHz']-1)})
    csvout(TAB/'bloch_final_changes.csv',changes,['branch_lower','lower_change_percent','upper_change_percent'])
    cav=json.loads((LOG/'stage_cavity.json').read_text());rob=json.loads((LOG/'stage_robustness.json').read_text());c=cav['best']
    freqs_converged=abs(c['percent_change_frequency_GHz'])<=2 and c['overlap_previous']>=.9
    loc_converged=max(abs(c['delta_eta_strain_1um']),abs(c['delta_eta_strain_2um']),abs(c['delta_eta_displacement_1um']),abs(c['delta_eta_displacement_2um']))<=.02 and c['overlap_previous']>=.9
    gap_pass=bool(changes) and max(abs(v) for r in changes for k,v in r.items() if k!='branch_lower')<=2
    status={'geometry':'PASS','FEM_numerical_tests':'PASS','harmonic_sweep':'PASS_NUMERICAL',
            'sampled_Bloch_mesh_convergence':'PASS' if gap_pass else 'NOT_ESTABLISHED',
            'best_tracked_cavity_frequency_convergence':'PASS' if freqs_converged else 'NOT_CONVERGED',
            'best_tracked_cavity_localization_convergence':'PASS' if loc_converged else 'NOT_CONVERGED',
            'finite_transmission_mesh_convergence':'PASS' if rob['all_transmission_mesh_checks_pass'] else 'FAIL',
            'finite_transmission_sponge_sensitivity':'PASS' if rob['all_sponge_checks_pass'] else 'FAIL',
            'HPGe_4K_material_validation':'BLOCKED_MISSING_VERIFIED_PROPERTIES',
            'specimen_orientation':'UNKNOWN_ASSUMED_100_ON_001','3D_validation':'NOT_RUN',
            'publication_ready_for_4K_HPGe_device_claims':False}
    js(TAB/'validation_status.json',status)
    thermal=n.thermal(c['frequency_GHz'],4)
    summary={'status':status,'best_candidate':c,'sampled_gaps':gaps10,'strongest_suppression_GHz':float(best['frequency_GHz']),'strongest_suppression_dB':float(best['defect_relative_dB']),'thermal_best_candidate':thermal,'sweep_frequencies':len(sweep),'folder':str(OUT)}
    js(OUT/'summary.json',summary)
    text='# 4 K HPGe-requested nanobeam study\n\n'
    text+='**Publication status: NOT READY for claims about a specific 4 K HPGe device.** The completed calculation uses a nominal cubic-Ge elastic proxy and a 4 K bath for thermal occupation. The numerical tensor is not temperature-calibrated. High purity, isotope composition, orientation, and GHz loss are unmeasured.\n\n'
    text+='## Model and inputs\n\nOriginal 26-hole workbook validated again; dimensions held fixed. Provisional width 1000 nm and exact-device leads 1500 nm. Cubic tensor C11=128.5, C12=48.3, C44=66.8 GPa, rho=5323 kg/m³; x=[100], y=[010], z=[001] is an assumption. Plane stress with unit thickness; no out-of-plane modes, intrinsic HPGe attenuation, thermal prestress, optical coupling or Q prediction. All previous simulation outputs were read-only.\n\n'
    text+='## Fresh calculations\n\n'
    text+=f'- 4 GHz validation and the five-frequency check passed; {len(sweep)} frequencies × three structures were solved.\n'
    text+=f'- Strongest sampled defect/reference displacement suppression: {float(best["frequency_GHz"]):.3f} GHz, {float(best["defect_relative_dB"]):.3f} dB. This is exploratory and must be read with the convergence and sponge checks. It is not transmitted power.\n'
    for g in gaps10:text+=f'- Sampled in-plane Bloch gap: {g["lower_GHz"]:.6f}–{g["upper_GHz"]:.6f} GHz (nominal cubic proxy, [100]).\n'
    if not gaps10:text+='- No sampled complete gap below 8 GHz among the computed branches.\n'
    text+=f'- {cav["analyzed_modes"]} finite free-ended cavity eigenmodes analyzed over three meshes; four physical identities tracked using FE shape interpolation, area-weighted overlap and frequency proximity.\n'
    text+=f'- Best tracked candidate: {c["frequency_GHz"]:.6f} GHz; in a sampled gap: {c["inside_gap"]}. Strain fractions within |x|<1/2/2.5 µm: {c["eta_strain_1um"]:.4f}/{c["eta_strain_2um"]:.4f}/{c["eta_strain_2p5um"]:.4f}. Displacement fractions: {c["eta_displacement_1um"]:.4f}/{c["eta_displacement_2um"]:.4f}/{c["eta_displacement_2p5um"]:.4f}. Maximum strain x={c["x_max_strain_um"]:.4f} µm.\n'
    text+=f'- At 4 K this candidate has equilibrium mean occupation {thermal["mean_occupation"]:.3f} and single-quantum energy {thermal["quantum_energy_microeV"]:.4f} µeV. These do not specify a classical source amplitude or measured readout power.\n\n'
    text+='## Validation gates\n\n'+''.join(f'- {k}: **{v}**\n' for k,v in status.items())
    text+='\n## Interpretation and limits\n\n'
    text+='The 4 K bath changes the Bose occupation calculation, not the classical displacement operator with fixed elastic inputs. A comparison against the old isotropic result changes the constitutive model and must not be described as a measured cooling shift. The [110] sensitivity is a second assumption, not a crystal-orientation measurement. Bloch gap edges use adaptive sampling of 16 computed in-plane branches; this is a numerical estimate, not proof of a full 3D bandgap. Coarse high-frequency transmission may be under-resolved. Central localization integrals use degree-five quadrature with finite-window sampling; mode-area is an inverse participation area. End-loss sensitivity tests use a weak mass sponge and cannot establish radiation-limited Q.\n\n'
    text+='Necessary publication work: verify low-temperature Cij and density including the Fine erratum; obtain crystal orientation, thickness and fabrication metrology; include cubic 3D modes or demonstrate the validity of the 2D reduction; establish converged energy-flux or port transmission with validated nonreflecting boundaries; characterize loss, thermal contraction and support stress.\n\n'
    text+='See MATERIAL_PROVENANCE.md for primary-source links. Figures are supplied as PNG and selected PDF files. Raw meshes, full selected complex fields, source hashes, validation tables and run logs are preserved.\n\n'
    text+='## Reproduction\n\nFrom this directory, using the existing local environment:\n\n```sh\npython run.py --output ../run_NEW_UNIQUE_ID --stage all\n```\n\nA new sibling output directory must not already exist. Completed output cannot be silently replaced. `--resume` explicitly continues a partially completed run and skips completed stages. The project workbook, original direct FEM module, and an interpreter with openpyxl are required; hashes and versions are recorded. No packages are installed.\n'
    (OUT/'SUMMARY.md').write_text(text)
    return summary


if __name__=='__main__':
    sentinel=LOG/'started.json'
    if (LOG/'complete.json').exists():raise SystemExit('Run is already complete; use --output with a new sibling directory.')
    if sentinel.exists() and not args.resume:raise SystemExit('Existing run; use --resume explicitly or a new --output directory.')
    if not sentinel.exists():js(sentinel,{'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),'scope_confirmed':'26-hole nanobeam','specimen_information':'user has none'})
    try:
        VARIANTS,BOUNDS=geometry();snapshot()
        stage_functions=[('smoke',smoke),('sweep',sweep),('bands',bands),('cavity',cavities),('robustness',robustness)]
        requested={'smoke':{'smoke'},'sweep':{'smoke','sweep'},'bands':{'smoke','bands'},
                   'cavity':{'smoke','bands','cavity'},'robustness':{'smoke','sweep','bands','robustness'},
                   'all':{name for name,_ in stage_functions}}[args.stage]
        for stage,fn in stage_functions:
            if stage not in requested:continue
            path=LOG/f'stage_{stage}.json'
            if path.exists():
                previous=json.loads(path.read_text())
                evidence=previous.get('_output_sha256') if isinstance(previous,dict) else None
                if not isinstance(evidence,dict) or not evidence:
                    raise ValueError('Stage marker lacks verified artifact hashes: '+stage)
                for rel,digest in evidence.items():
                    artifact=OUT/rel
                    if not artifact.is_file() or sha(artifact)!=digest:
                        raise ValueError('Completed-stage artifact missing/changed: '+rel+'; use a new run')
                log('Skip verified completed stage',stage);continue
            js(LOG/'status.json',{'state':'running','stage':stage,'pid':os.getpid()});log('START',stage)
            result=fn()
            result['_output_sha256']={str(p.relative_to(OUT)):sha(p) for group in (TAB,FLD,MESH,FIG) for p in sorted(group.rglob('*')) if p.is_file()}
            js(path,result);log('COMPLETE',stage)
        if all((LOG/f'stage_{name}.json').exists() for name,_ in stage_functions):
            result=report();js(LOG/'complete.json',{'status':'COMPLETE_CONDITIONAL_STUDY','publication_ready':False})
            js(LOG/'status.json',{'state':'complete','publication_ready':False})
        else:
            js(LOG/'status.json',{'state':'selected_stages_complete','requested_stage':args.stage,'publication_ready':False})
        log('SELECTED STAGES COMPLETE',str(OUT))
    except Exception as exc:
        details=traceback.format_exc();log(details);js(LOG/'status.json',{'state':'failed','error':str(exc)});(LOG/'failure.txt').write_text(details);raise
