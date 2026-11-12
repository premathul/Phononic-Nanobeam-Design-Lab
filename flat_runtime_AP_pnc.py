"""Flat distribution adapter. Original solver bytes and paths are restored temporarily."""
import hashlib,json,shutil,subprocess,sys,tempfile
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_manifest(path):return json.loads(path.read_text())
def restore(rows,dest):
 for row in rows:
  rel=Path(row['original']);src=ROOT/row['flat']
  if rel.is_absolute() or '..' in rel.parts or Path(row['flat']).name!=row['flat']:raise ValueError('Unsafe manifest path')
  if digest(src)!=row['sha256']:raise ValueError('File changed: '+row['flat'])
  out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
@contextmanager
def project():
 with tempfile.TemporaryDirectory(prefix='athul_pnc_work_') as td:
  dest=Path(td)/'project';dest.mkdir();restore(load_manifest(ROOT/'flat_manifest.json'),dest)
  yield dest

def collect(out,name):
 rows=[]
 if out.exists():
  for src in sorted(out.rglob('*')):
   if not src.is_file():continue
   rel=src.relative_to(out).as_posix();flat='new_'+name+'__'+rel.replace('/','__')
   if flat.endswith('.py'):flat=flat[:-3]+'_AP_pnc.py'
   target=ROOT/flat;shutil.copy2(src,target)
   rows.append({'original':rel,'flat':flat,'sha256':digest(target)})
  target=ROOT/('new_'+name+'__manifest.json');tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(rows,indent=2)+'\n');tmp.replace(target)
