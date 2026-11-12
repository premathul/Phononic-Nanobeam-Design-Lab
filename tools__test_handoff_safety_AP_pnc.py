#!/usr/bin/env python3
"""Small no-physics checks of launcher/output protections."""
import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/'results/4K/portable_template/run.py'
def command(*args):
 return subprocess.run([sys.executable,'-B',*map(str,args)],cwd=ROOT,text=True,capture_output=True)
class SafetyTests(unittest.TestCase):
 def test_dry_run_does_not_create_output(self):
  label='dryrun_validation_349591';out=ROOT/'results/4K'/('run_new_'+label)
  self.assertFalse(out.exists())
  p=command(ROOT/'handoff.py','--name',label,'--dry-run')
  self.assertEqual(p.returncode,0,p.stderr);self.assertFalse(out.exists())
 def test_path_name_rejected(self):
  p=command(ROOT/'handoff.py','--name','../unsafe')
  self.assertNotEqual(p.returncode,0)
 def test_archived_run_rejected(self):
  archived=ROOT/'results/4K/run_20261011T002256Z'
  p=command(RUNNER,'--output',archived,'--resume')
  self.assertNotEqual(p.returncode,0);self.assertIn('never the template or archived baseline',p.stderr)
 def test_missing_output_option_rejected(self):
  p=command(RUNNER)
  self.assertNotEqual(p.returncode,0);self.assertIn('--output',p.stderr)
 def test_existing_lock_preserved(self):
  with tempfile.TemporaryDirectory(prefix='run_new_safety_',dir=ROOT/'results/4K') as td:
   out=Path(td);lock=out/'.run.lock';content='{"pid":0,"token":"test-only-existing-lock"}'
   lock.write_text(content)
   p=command(RUNNER,'--output',out,'--resume')
   self.assertNotEqual(p.returncode,0);self.assertIn('Run lock exists',p.stderr)
   self.assertEqual(lock.read_text(),content)
   self.assertEqual(sorted(x.name for x in out.iterdir()),['.run.lock'])
 def test_numerics_unchanged(self):
  a=ROOT/'results/4K/run_20261011T002256Z/numerics.py'
  b=ROOT/'results/4K/portable_template/numerics.py'
  self.assertEqual(a.read_bytes(),b.read_bytes())
if __name__=='__main__':unittest.main(verbosity=2)
