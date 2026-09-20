"""Preflight without loading a second 7B model."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import importlib.metadata as m
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--cpu-only',action='store_true');a=p.parse_args()
print('Python',sys.version)
for line in (ROOT/'requirements.txt').read_text().splitlines():
 name,want=line.split('==');got=m.version(name)
 if got!=want: raise SystemExit(f'Version mismatch {name}: want {want}, got {got}')
 print(name,got)
import torch
if torch.__version__!='2.5.1+cu121':raise SystemExit('Expected torch 2.5.1+cu121; use the documented environment for matched comparisons')
for name,want in json.loads((ROOT/'scripts/source_checksums.json').read_text()).items():
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want, f'Source changed: {name}'
checks=json.loads((ROOT/'scripts/data_checksums.json').read_text())
for name,want in checks.items():assert hashlib.sha256((ROOT/'safecomp_audit/data/mgsm'/name).read_bytes()).hexdigest()==want
sys.path.insert(0,str(ROOT/'safecomp_audit/src'))
from mgsm_grpo import load_parallel
per,tr,ev=load_parallel(['bn','te','th'])
assert len(tr)==190 and len(ev)==60 and not set(tr)&set(ev)
assert all(len(v)==250 for v in per.values())
assert all(len({v[i]['gold'] for v in per.values()})==1 for i in range(250))
print('Data PASS: identical gold answers aligned over 250 IDs; 190 train / 60 held out; default evaluation uses first 50 held out.')
if not a.cpu_only:
 assert torch.cuda.is_available()
 print(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free','--format=csv,noheader'],text=True))
 assert torch.cuda.get_device_properties(a.gpu).major>=8,'BF16-capable Ampere or newer expected'
print('PASS')
