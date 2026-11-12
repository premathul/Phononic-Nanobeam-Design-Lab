#!/usr/bin/env python3
"""Read-only checks of source hashes and saved result arithmetic; no simulation."""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/4K/run_20261011T002256Z'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def rows(name,base=BASE):
 with (base/'tables'/name).open(newline='') as f:return list(csv.DictReader(f))
def require(condition,message):
 if not condition:raise ValueError(message)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--fields',action='store_true',help='Also check saved NPZ arrays (requires NumPy)')
 p.add_argument('--compare-run',type=Path,help='New run whose five-frequency gate should match archived baseline')
 a=p.parse_args()
 hashes=json.loads((ROOT/'provenance/copied_original_sha256.json').read_text())
 for rel,digest in hashes.items():require(sha(ROOT/rel)==digest,'Copied original differs: '+rel)
 saved=json.loads((BASE/'logs/artifact_manifest_sha256.json').read_text())
 omitted=[]
 for rel,digest in saved.items():
  if Path(rel).name=='.DS_Store' or any(part in ('cache','matplotlib-cache','__pycache__') for part in Path(rel).parts):
   omitted.append(rel);continue
  require(sha(BASE/rel)==digest,'Archived artifact differs: '+rel)
 r=rows('frequency_sweep.csv');require(len(r)==111,'Expected 111 sweep rows')
 max_res=max_balance=0.
 for i,row in enumerate(r):
  require(abs(float(row['frequency_GHz'])-(1+.1*i))<1e-10,'Frequency grid mismatch')
  for kind in ('reference','periodic','defect'):
   vals=[float(row[kind+'_'+key]) for key in ('residual','power_balance_relative_error','P1_u_m','P3_u_m')]
   require(all(math.isfinite(v) for v in vals),'Nonfinite sweep value')
   require(0<=vals[0]<1e-7 and 0<=vals[1]<1e-6 and min(vals[2:])>0,'Numerical residual/probe gate failed')
   max_res=max(max_res,vals[0]);max_balance=max(max_balance,vals[1])
   if kind!='reference':
    expected=20*math.log10(float(row[kind+'_P3_u_m'])/float(row['reference_P3_u_m']))
    require(abs(expected-float(row[kind+'_relative_dB']))<1e-10,'Stored transmission arithmetic differs')
 modes=rows('all_cavity_candidates.csv');require(len(modes)==60,'Expected 60 cavity modes')
 require(all(0<=float(x['eigen_relative_residual'])<1e-7 for x in modes),'Cavity eigen residual failed')
 status=json.loads((BASE/'tables/validation_status.json').read_text())
 require(status['finite_transmission_mesh_convergence']=='FAIL','Preserve failed mesh gate')
 require(status['finite_transmission_sponge_sensitivity']=='FAIL','Preserve failed sponge gate')
 require(status['publication_ready_for_4K_HPGe_device_claims'] is False,'Do not promote validation status')
 result={'status':'PASS_SAVED_DATA_CHECKS','copied_original_hashes_checked':len(hashes),
         'archived_manifest_files_checked':len(saved)-len(omitted),'excluded_metadata_or_cache':omitted,
         'harmonic_solves_checked':333,'cavity_eigen_residuals_checked':60,
         'maximum_harmonic_residual':max_res,'maximum_power_balance_error':max_balance,
         'physical_validation_failures_preserved':True,'simulation_rerun':False}
 if a.fields:
  import numpy as np
  count=0
  for path in (BASE/'fields').glob('*.npz'):
   with np.load(path,allow_pickle=False) as z:
    for key in z.files:require(np.isfinite(z[key]).all(),'Nonfinite NPZ data: '+path.name+'/'+key)
    points=z['points_m'];tri=z['triangles']
    require(tri.ndim==2 and tri.shape[1]==3 and tri.min()>=0 and tri.max()<len(points),'Invalid triangles')
    if 'displacement_complex' in z:
     require(len(z['displacement_complex'])==2*len(points),'Displacement shape mismatch')
     require(len(z['strain_energy_density'])==len(tri),'Strain shape mismatch')
    if 'modes' in z:require(z['modes'].shape[0]==2*len(points),'Eigenvector shape mismatch')
   count+=1
  result['saved_npz_files_checked']=count
 if a.compare_run:
  marker=json.loads((a.compare_run.resolve()/'logs/stage_smoke.json').read_text())
  require(all(marker.get(k)=='PASS' for k in ('four_GHz','five_frequencies','operator_regression')),'Smoke/operator success marker missing')
  got=rows('five_frequency_comparison.csv',a.compare_run.resolve());expected=rows('five_frequency_comparison.csv')
  require(len(got)==len(expected)==5,'Expected five gate rows')
  differences=[]
  for x,y in zip(got,expected):
   require(float(x['frequency_GHz'])==float(y['frequency_GHz']),'Gate frequencies differ')
   for kind in ('reference','periodic','defect'):
    residual=float(x[kind+'_residual']); balance=float(x[kind+'_power_balance_relative_error'])
    probes=[float(x[kind+'_'+k]) for k in ('P1_u_m','P3_u_m')]
    require(math.isfinite(residual) and 0<=residual<1e-7,'Fresh harmonic residual failed')
    require(math.isfinite(balance) and 0<=balance<1e-6,'Fresh balance failed')
    require(all(math.isfinite(v) and v>0 for v in probes),'Fresh probe invalid')
   for kind in ('defect','periodic'):
    db=float(x[kind+'_relative_dB']); olddb=float(y[kind+'_relative_dB'])
    recomputed=20*math.log10(float(x[kind+'_P3_u_m'])/float(x['reference_P3_u_m']))
    require(math.isfinite(db) and math.isfinite(olddb) and abs(db-recomputed)<1e-10,'Fresh dB invalid')
    differences.append(abs(db-olddb))
  require(max(differences)<=.02,'Fresh gate differs by >0.02 dB; investigate environment/mesh')
  result['fresh_five_frequency_max_difference_dB']=max(differences)
  result['fresh_smoke_comparison']='PASS'
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:
  print(json.dumps({'status':'FAIL','error':str(e)},indent=2));sys.exit(1)
