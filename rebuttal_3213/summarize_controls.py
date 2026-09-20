"""Summarize complete, prospectively specified control cohorts. Fails on missing runs.
Uses original problem-ID clusters, keeping all languages together. Exploratory CI.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'runs')
p.add_argument('--arms',default='vanilla,split,cgh_strict,cgh_strict_nois,pivot_rsft')
p.add_argument('--seeds',default='1,2,3'); p.add_argument('--steps',type=int,default=100)
a=p.parse_args(); seeds=[int(s) for s in a.seeds.split(',')]; arms=a.arms.split(',')
data={}; hashes={}; protocols=[]
for arm in arms:
 for s in seeds:
  directory=a.root/f'{arm}_s{s}_n{a.steps}'
  f=directory/'result.json'
  if not f.exists(): raise SystemExit(f'Missing requested run: {f}; do not select a different seed based on outcomes.')
  d=json.loads(f.read_text()); cfg=json.loads((directory/'config.json').read_text())
  assert d['steps']==a.steps and d['seed']==s and d['arm']==arm
  hashes[(arm,s)]=cfg['initial_lora_sha256']
  protocols.append({k:cfg[k] for k in ('source_sha256','coefficients_sha256','torch','transformers','peft','train_ids','eval_ids','dropout','max_new','extra','eval_batch','entropy_alpha')})
  assert (Path(cfg['model']).name == 'a09a35458c702b33eeacc393d103063234e8bc28' or cfg['model']=='Qwen/Qwen2.5-7B-Instruct'), 'Unknown model; verify revision before merging' # Hub-ID form only for audited source-server run (HANDOFF.md)
  data[(arm,s)]={(lg,r['i']):r for lg,v in d['eval_end'].items() for r in v['per_problem']}
assert all(v==protocols[0] for v in protocols), 'Protocol/environment mismatch across runs; do not pool these runs'
for s in seeds: assert len({hashes[(arm,s)] for arm in arms})==1, f'Initialization mismatch, seed {s}'
keys=sorted(data[(arms[0],seeds[0])]); assert all(sorted(v)==keys for v in data.values())
ids=sorted({i for lg,i in keys}); langs=sorted({lg for lg,i in keys})
rng=np.random.default_rng(3213)
summary={'seeds':seeds,'distinct_semantic_items':len(ids),'languages':langs,'arms':{},'comparisons':[]}
for arm in arms:
 summary['arms'][arm]={f:float(np.mean([r[f] for s in seeds for r in data[(arm,s)].values()])) for f in ('corr','corr_raw','lang')}
for left,right in [('vanilla','split'),('split','cgh_strict'),('pivot_rsft','cgh_strict'),('cgh_strict_nois','cgh_strict'),('q_resample','cgh_strict'),('absolute','cgh_strict'),('whole_gate','cgh_strict'),('component_zvp','cgh_strict'),('primary_reward','cgh_strict'),('dynamic_correctness','cgh_strict'),('split','cgh_noparallel'),('cgh_legacy','cgh_strict')]:
 if left not in arms or right not in arms: continue
 delta=np.array([[np.mean([data[(right,s)][(l,i)]['corr']-data[(left,s)][(l,i)]['corr'] for l in langs]) for i in ids] for s in seeds])
 ds=delta.mean(axis=1); nz=ds[np.abs(ds)>1e-12]; samples=[]
 for _ in range(20000): samples.append(delta[np.ix_(rng.integers(len(seeds),size=len(seeds)),rng.integers(len(ids),size=len(ids)))].mean())
 summary['comparisons'].append({'left':left,'right':right,'delta':float(ds.mean()),'per_seed':ds.tolist(),'exploratory_crossed_95ci':np.quantile(samples,[.025,.975]).tolist(),'sign_p_two_sided':float(binomtest(int((nz>0).sum()),len(nz)).pvalue) if len(nz) else 1.})
print(json.dumps(summary,indent=2))
(a.root/'summary.json').write_text(json.dumps(summary,indent=2))
