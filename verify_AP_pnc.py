#!/usr/bin/env python3
"""Verify flat hashes and original scientific-result gates, without running physics."""
import argparse,subprocess,sys
sys.dont_write_bytecode=True
from flat_runtime_AP_pnc import ROOT,project,restore,load_manifest
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fields',action='store_true');p.add_argument('--compare-name');p.add_argument('--safety',action='store_true');a=p.parse_args()
with project() as work:
 cmd=[sys.executable,'-B',str(work/'tools/verify_saved_results.py')]
 if a.fields:cmd+=['--fields']
 if a.compare_name:
  out=work/'results/4K'/('run_new_'+a.compare_name);restore(load_manifest(ROOT/('new_'+a.compare_name+'__manifest.json')),out);cmd+=['--compare-run',str(out)]
 code=subprocess.call(cmd,cwd=work)
 if code==0 and a.safety:code=subprocess.call([sys.executable,'-B',str(work/'tools/test_handoff_safety.py')],cwd=work)
 sys.exit(code)
