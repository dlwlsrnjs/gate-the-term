"""Fetch hash-verified MGSM files and the model revision used on the source server."""
import argparse,hashlib,json,os,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REV='a09a35458c702b33eeacc393d103063234e8bc28'
p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true');a=p.parse_args()
checks=json.loads((ROOT/'scripts/data_checksums.json').read_text())
out=ROOT/'safecomp_audit/data/mgsm';out.mkdir(parents=True,exist_ok=True)
for name,want in checks.items():
 f=out/name
 b=f.read_bytes() if f.exists() else urllib.request.urlopen('https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/'+name,timeout=60).read()
 if hashlib.sha256(b).hexdigest()!=want: raise SystemExit(f'Checksum mismatch: {name}. Refusing changed data.')
 if not f.exists(): f.write_bytes(b)
 print('Verified',name)
if not a.data_only:
 os.environ.setdefault('HF_HOME',str(Path.home()/'.cache/huggingface'))
 from huggingface_hub import snapshot_download
 path=snapshot_download('Qwen/Qwen2.5-7B-Instruct',revision=REV)
 (ROOT/'scripts/model_path.txt').write_text(path+'\n')
 print('Model ready:',path)
