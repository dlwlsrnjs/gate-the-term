"""Portable, sequential per-GPU launcher. Complete jobs skip; incomplete jobs fail closed."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rebuttal_3213'))
from train_controls import ARMS
PRIMARY=['vanilla','split','cgh_strict','pivot_rsft','cgh_strict_nois']
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu',type=int,default=0);p.add_argument('--seeds',default='1,2,3')
p.add_argument('--arms',default=','.join(PRIMARY));p.add_argument('--steps',type=int,default=100)
p.add_argument('--eval-n',type=int,default=50);p.add_argument('--output',type=Path,default=ROOT/'outputs/controls')
p.add_argument('--model',default=None);p.add_argument('--min-free-mib',type=int,default=24000)
p.add_argument('--eval-batch',type=int,default=8);p.add_argument('--dry-run',action='store_true')
a=p.parse_args();a.output=a.output.resolve();arms=a.arms.split(',');seeds=[int(x) for x in a.seeds.split(',')]
if len(set(arms))!=len(arms) or not set(arms)<=set(ARMS) or len(set(seeds))!=len(seeds): p.error('Unique valid arms and seeds required')
if a.steps<1 or not 1<=a.eval_n<=60: p.error('steps >= 1; eval-n in 1..60')
model_file=ROOT/'scripts/model_path.txt'
model=a.model or (model_file.read_text().strip() if model_file.exists() else None)
if not model and not a.dry_run: p.error('Run scripts/prepare.py first, or pass --model /absolute/snapshot/path')
model=model or 'Qwen/Qwen2.5-7B-Instruct'
# Set portable cache before loading any model; trainer preserves supplied env.
os.environ.setdefault('HF_HOME',str(Path.home()/'.cache/huggingface'))
a.output.mkdir(parents=True,exist_ok=True)
lock=(a.output/f'.gpu{a.gpu}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
source=hashlib.sha256((ROOT/'rebuttal_3213/train_controls.py').read_bytes()).hexdigest()
coeff=hashlib.sha256((ROOT/'rebuttal_3213/control_coefficients.py').read_bytes()).hexdigest()
manifest={**vars(a),'model':model,'arms':arms,'seeds':seeds,'source_sha256':source,'coefficients_sha256':coeff,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(R:=a.output/f'manifest_gpu{a.gpu}_{time.time_ns()}.json').write_text(json.dumps(manifest,default=str,indent=2))
for seed in seeds:
 for arm in arms:
  tag=f'{arm}_s{seed}_n{a.steps}';dest=a.output/tag
  cmd=[sys.executable,'-u',str(ROOT/'rebuttal_3213/train_controls.py'),'--gpu',str(a.gpu),'--arm',arm,'--seed',str(seed),'--steps',str(a.steps),'--eval-n',str(a.eval_n),'--eval-batch',str(a.eval_batch),'--model',model,'--min-free-mib',str(a.min_free_mib),'--output',str(a.output)]
  if a.dry_run: print(' '.join(cmd));continue
  joblock=(a.output/f'.{tag}.lock').open('w');fcntl.flock(joblock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if dest.exists():
   if not (dest/'result.json').exists(): raise SystemExit(f'Incomplete run: {dest}. Archive it outside the output directory before restarting this job. No optimizer/RNG resume is supported.')
   d=json.loads((dest/'result.json').read_text());c=json.loads((dest/'config.json').read_text())
   expected={'arm':arm,'seed':seed,'steps':a.steps,'eval_n':a.eval_n,'model':model,'source_sha256':source,'coefficients_sha256':coeff,'eval_batch':a.eval_batch,'extra':2,'max_new':320,'entropy_alpha':1.}
   if any(c.get(k)!=v for k,v in expected.items()):raise SystemExit(f'Configuration mismatch: {dest}; use a separate output root')
   if any(d.get(k)!=expected[k] for k in ('arm','seed','steps')) or set(d.get('eval_end',{}))!={'bn','te','th'}:raise SystemExit(f'Invalid result: {dest}')
   if any(len(v['per_problem'])!=a.eval_n for v in d['eval_end'].values()):raise SystemExit(f'Incomplete evaluation: {dest}')
   print('SKIP complete',tag,flush=True);joblock.close();continue
  state={'job':tag,'state':'running','gpu':a.gpu,'command':cmd}
  status=a.output/f'status_gpu{a.gpu}.json';status.write_text(json.dumps(state,indent=2))
  print('START',tag,flush=True)
  with (a.output/f'{tag}.log').open('w') as log: rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
  state.update(state='complete' if rc==0 and (dest/'result.json').exists() else 'failed',returncode=rc)
  status.write_text(json.dumps(state,indent=2));joblock.close()
  if state['state']=='failed':raise SystemExit(f'Failed {tag}; inspect {a.output}/{tag}.log. Queue stopped.')
  print('DONE',tag,flush=True)
print('Requested jobs complete' if not a.dry_run else 'Dry run complete',flush=True)
