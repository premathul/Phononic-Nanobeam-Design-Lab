#!/usr/bin/env python3
"""Run original staged FEM through a flat-folder adapter. Smoke is the default."""
import argparse,json,os,re,subprocess,sys
sys.dont_write_bytecode=True
from flat_runtime_AP_pnc import ROOT,project,restore,load_manifest,collect
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--name',required=True);p.add_argument('--stage',choices=['smoke','sweep','bands','cavity','robustness','all'],default='smoke')
p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--excel-python')
p.add_argument('--postprocess',choices=['verify_eigen_scaling','check_legacy_bloch','finalize'])
a=p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',a.name):p.error('Use a unique simple label of up to 40 characters')
manifest=ROOT/('new_'+a.name+'__manifest.json')
if a.postprocess and not a.resume:p.error('--postprocess requires --resume')
if a.resume and not manifest.exists():p.error('No collected run to resume')
if not a.resume and (manifest.exists() or any(ROOT.glob('new_'+a.name+'__*'))):p.error('Run name already exists')
if a.dry_run:
 print(json.dumps({'stage':a.stage,'output_prefix':'new_'+a.name+'__','temporary_working_tree':True,'solver':'portable__run_AP_pnc.py','resume':a.resume,'postprocess':a.postprocess},indent=2));sys.exit(0)
lock=ROOT/('new_'+a.name+'__flat.lock')
try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
except FileExistsError:p.error('Flat run lock exists. Verify no process is running before removing it.')
os.write(fd,str(os.getpid()).encode());os.close(fd)
try:
 with project() as work:
  out=work/'results/4K'/('run_new_'+a.name)
  if a.resume:restore(load_manifest(manifest),out)
  if a.postprocess:cmd=[sys.executable,'-B',str(out/(a.postprocess+'.py'))]
  else:
   cmd=[sys.executable,'-B',str(work/'handoff.py'),'--name',a.name,'--stage',a.stage]
   if a.resume:cmd+=['--resume']
   if a.excel_python:cmd+=['--excel-python',a.excel_python]
  try:code=subprocess.call(cmd,cwd=work)
  finally:collect(out,a.name)
finally:lock.unlink(missing_ok=True)
sys.exit(code)
