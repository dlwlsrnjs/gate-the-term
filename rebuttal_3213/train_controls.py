"""Prospective real-model controls. Does not overwrite submitted code/results.
All arms draw G original + K extra responses per prompt (fixed response-count cap).
Extra conditioning: q for q_resample; English-augmented q otherwise.
All evaluation uses q only. Count/token budgets are logged, not conflated.
This is a clean rerun with seeded LoRA initialization and dropout disabled, NOT
bitwise reproduction of historical runs. No arm is labelled DAPO or RL-ZVP.
"""
import argparse, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'safecomp_audit'
ARMS=['vanilla','split','cgh_legacy','cgh_strict','cgh_strict_nois','pivot_rsft','q_resample','whole_gate','absolute','component_zvp','primary_reward','cgh_noparallel','dynamic_correctness']
def arguments():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--arm',choices=ARMS,required=True); p.add_argument('--seed',type=int,default=1)
 p.add_argument('--steps',type=int,default=100); p.add_argument('--eval-n',type=int,default=50)
 p.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct'); p.add_argument('--gpu',type=int,default=0)
 p.add_argument('--eval-batch',type=int,default=8)
 p.add_argument('--entropy-alpha',type=float,default=1.)
 p.add_argument('--diagnostic-adapter',type=Path)
 p.add_argument('--diagnostic-prompts',type=int,default=0)
 p.add_argument('--diagnostic-draws',type=int,default=16)
 p.add_argument('--max-new',type=int,default=320); p.add_argument('--extra',type=int,default=2)
 p.add_argument('--min-free-mib',type=int,default=24000); p.add_argument('--dry-run',action='store_true')
 p.add_argument('--output',type=Path,default=HERE/'runs'); p.add_argument('--eval-only-adapter',type=Path)
 return p.parse_args()
def main():
 a=arguments()
 source_snapshot=Path(__file__).read_text()
 coeff_snapshot=(HERE/'control_coefficients.py').read_text()
 if a.eval_n>60 or a.eval_n<1: raise ValueError('Only 60 disjoint held-out semantic items exist; do not evaluate training items.')
 if a.extra<1: raise ValueError('extra must be positive')
 if a.dry_run:
  print(json.dumps(vars(a),default=str,indent=2)); return
 free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
 if int(free[a.gpu])<a.min_free_mib: raise RuntimeError(f'GPU {a.gpu}: {free[a.gpu]} MiB free; need {a.min_free_mib}. No training started.')
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 os.environ.setdefault('HF_HUB_OFFLINE','1')
 os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
 os.environ.setdefault('HF_HOME','/home/ubuntu/342/jinkwon/.cache/huggingface')
 os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
 import numpy as np
 import torch
 import torch.nn.functional as F
 import transformers, peft
 from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
 from peft import LoraConfig, get_peft_model, PeftModel
 sys.path.insert(0,str(ROOT/'src'))
 import mgsm_grpo as old
 torch.set_num_threads(4)
 set_seed(a.seed) # before LoRA construction; old trainer did this after construction
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='left')
 if tok.pad_token_id is None: tok.pad_token=tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
 if a.eval_only_adapter or a.diagnostic_adapter:
  model=PeftModel.from_pretrained(base,str(a.eval_only_adapter or a.diagnostic_adapter),is_trainable=bool(a.diagnostic_adapter)).to('cuda')
 else:
  model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])).to('cuda')
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  model.enable_input_require_grads()
 for module in model.modules():
  if isinstance(module,torch.nn.Dropout): module.p=0.
 model.config.use_cache=False
 per,train,ev=old.load_parallel(['bn','te','th'])
 assert not set(train)&set(ev)
 initial=hashlib.sha256()
 for n,p in model.named_parameters():
  if 'lora_' in n: initial.update(p.detach().float().cpu().numpy().tobytes())
 tag=f'{a.arm}_s{a.seed}_n{a.steps}'
 if a.diagnostic_prompts: tag=f'diagnostic_s{a.seed}_p{a.diagnostic_prompts}_k{a.diagnostic_draws}'
 if a.eval_only_adapter: tag=f'eval_{a.eval_only_adapter.name}_n{a.eval_n}'
 out=a.output/tag; out.mkdir(parents=True,exist_ok=False)
 cfg={**vars(a),'initial_lora_sha256':initial.hexdigest(),'source_sha256':hashlib.sha256(source_snapshot.encode()).hexdigest(),'train_ids':train,'eval_ids':ev[:a.eval_n],'dropout':0.,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'coefficients_sha256':hashlib.sha256(coeff_snapshot.encode()).hexdigest()}
 (out/'trainer_snapshot.py').write_text(source_snapshot)
 (out/'coefficients_snapshot.py').write_text(coeff_snapshot)
 (out/'config.json').write_text(json.dumps(cfg,default=str,indent=2))
 def prompt(q): return old.chat(tok,q)
 @torch.no_grad()
 def generate(q,n,greedy=False):
  model.eval(); enc=tok(prompt(q),return_tensors='pt').to('cuda')
  kw={'repetition_penalty':1.}
  if not greedy: kw.update(temperature=1.,top_p=1.,top_k=0)
  seq=model.generate(**enc,max_new_tokens=a.max_new,do_sample=not greedy,num_return_sequences=n,pad_token_id=tok.pad_token_id,use_cache=True,**kw)
  ans=[]
  for r in seq[:,enc.input_ids.shape[1]:]:
   rr=r.tolist()
   stop=model.generation_config.eos_token_id
   stop=set(stop if isinstance(stop,list) else [stop])
   first=next((j for j,t in enumerate(rr) if t in stop),None)
   if first is not None: rr=rr[:first+1]
   ans.append({'ids':rr,'text':tok.decode(rr,skip_special_tokens=True)})
  return ans,enc.input_ids.shape[1]*n
 def lp(q,r,grad=False,token_entropy=False):
  model.train(grad)
  pi=tok(prompt(q),return_tensors='pt').input_ids.to('cuda')
  ci=torch.tensor([r['ids']],device='cuda')
  with torch.set_grad_enabled(grad):
   ids=torch.cat([pi,ci],1)
   logits=model(ids,use_cache=False).logits[:,pi.shape[1]-1:-1]
   score=-F.cross_entropy(logits.transpose(1,2).float(),ci,reduction='none')
   if token_entropy:
    with torch.no_grad():
     # Chunk along time to avoid an extra full-vocabulary tensor for every token.
     ent=[]
     for chunk in logits.detach().float().split(32,dim=1):
      logz=torch.logsumexp(chunk,dim=-1)
      ent.append(logz-(torch.softmax(chunk,dim=-1)*chunk).sum(dim=-1))
     entropy=torch.cat(ent,dim=1)
    return score,entropy
   return score.mean()
 def norm(x):
  x=np.asarray(x,dtype=float); return (x-x.mean())/(x.std()+1e-4)
 def rewards(rs,lg,gold):
  for r in rs: r['c']=old.components(r['text'],gold,lg)
 def vals(rs,k): return np.array([r['c'][k] for r in rs])
 def aux(rs): return np.array([sum(w*r['c'][k] for k,w in old.WEIGHTS.items() if k!='corr') for r in rs])
 def scalar(rs): return norm([old.reward(r['c']) for r in rs])
 def split(rs): return norm(aux(rs))+norm(vals(rs,'corr'))
 @torch.no_grad()
 def evaluate():
  model.eval(); res={}
  for lg in ('bn','te','th'):
   rr=[]; ix=ev[:a.eval_n]
   for st in range(0,len(ix),a.eval_batch):
    chunk=ix[st:st+a.eval_batch]
    enc=tok([prompt(per[lg][i]['q']) for i in chunk],padding=True,return_tensors='pt').to('cuda')
    seq=model.generate(**enc,max_new_tokens=a.max_new,do_sample=False,repetition_penalty=1.,pad_token_id=tok.pad_token_id,use_cache=True)
    texts=tok.batch_decode(seq[:,enc.input_ids.shape[1]:],skip_special_tokens=True)
    for i,text in zip(chunk,texts):
     c=old.components(text,per[lg][i]['gold'],lg); rr.append({'i':i,**c,'text':text})
   res[lg]={'per_problem':rr,'n':len(rr),'acc':sum(r['corr'] for r in rr)/len(rr),'lang_consistent':sum(r['lang'] for r in rr)/len(rr)}
   (out/f'eval_{lg}.json').write_text(json.dumps(res[lg],ensure_ascii=False,indent=2))
   print('evaluated',lg,len(rr),flush=True)
  return res
 if a.eval_only_adapter:
  (out/'result.json').write_text(json.dumps({'eval_end':evaluate()},ensure_ascii=False,indent=2)); print('DONE',out,flush=True); return
 if a.diagnostic_prompts:
  # Finite-sample IS comparison in ONE LoRA block, never called a true-gradient measurement.
  candidates=[(n,p) for n,p in model.named_parameters() if '.self_attn.q_proj.lora_B.' in n]
  pname,param=candidates[-1]; records=[]
  def cosine(x,y):
   den=np.linalg.norm(x)*np.linalg.norm(y)
   return float(np.dot(x,y)/den) if den>0 else None
  for z in range(a.diagnostic_prompts):
   lg=('bn','te','th')[z%3]; i=train[z//3]; q=per[lg][i]['q']
   qp=old.PIVOT_TMPL.format(q=q,en=per['en'][i]['q'])
   torch.manual_seed(a.seed*1000000+z)
   extra,_=generate(qp,a.diagnostic_draws); rewards(extra,lg,per[lg][i]['gold'])
   logs=[]; grads=[]; lengths=[]; yy=[]
   for r in extra:
    lq=float(lp(q,r)); lt=float(lp(qp,r)); logs.append((lq-lt)*len(r['ids']))
    gm=torch.autograd.grad(lp(q,r,True),param)[0].detach().float().cpu().numpy().ravel()
    grads.append(gm); lengths.append(len(r['ids'])); yy.append(r['c']['corr'])
   logs=np.array(logs); lengths=np.array(lengths); yy=np.array(yy); gm=np.array(grads,dtype=np.float64)
   shifted=np.exp(logs-logs.max()); rootw=np.exp(np.clip(logs/lengths,np.log(.5),np.log(2.)))
   gs=gm*lengths[:,None]
   exact=(shifted[:,None]*yy[:,None]*gs).mean(0)
   root_seq=(rootw[:,None]*yy[:,None]*gs).mean(0)
   deployed=(rootw[:,None]*yy[:,None]*gm).mean(0)
   unweighted=(yy[:,None]*gm).mean(0)
   positive=shifted*yy
   row={'lang':lg,'i':i,'parameter_block':pname,'draws':len(extra),'successes':int(yy.sum()),'log_exact_ratios':logs.tolist(),'lengths':lengths.tolist(),'root_weights':rootw.tolist(),'exact_sample_ess':float(shifted.sum()**2/np.square(shifted).sum()),'success_weight_ess':float(positive.sum()**2/np.square(positive).sum()) if positive.sum()>0 else 0.,'cosine_exact_to_root_sequence':cosine(exact,root_seq),'cosine_exact_to_deployed':cosine(exact,deployed),'cosine_exact_to_unweighted':cosine(exact,unweighted),'root_clip_fraction':float(((logs/lengths<np.log(.5))|(logs/lengths>np.log(2.))).mean()),'exact_clip_fraction':float(((logs<np.log(.5))|(logs>np.log(2.))).mean())}
   records.append(row)
   (out/'diagnostics.json').write_text(json.dumps({'scope':'Single LoRA block; finite proposal-sample estimates, NOT full or true population gradients. Low ESS or zero successes makes direction unreliable.','records':records},indent=2))
   print(json.dumps({k:v for k,v in row.items() if not isinstance(v,list)}),flush=True)
  (out/'result.json').write_text(json.dumps({'diagnostic_complete':True,'records':records},indent=2)); print('DONE',out,flush=True); return
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
 rng=random.Random(a.seed); hist=[]; started=time.time()
 for step in range(a.steps):
  opt.zero_grad(set_to_none=True)
  for b in range(2):
   lg=rng.choice(['bn','te','th']); i=rng.choice(train); ex=per[lg][i]; q=ex['q']
   # Reset generation stream per prompt to avoid extra-arm RNG consumption changing later prompts.
   torch.manual_seed(a.seed*1000000+step*2+b)
   rs,pt=generate(q,4); rewards(rs,lg,ex['gold']); y=vals(rs,'corr'); deg=bool(y.max()==y.min())
   if a.arm in ('q_resample','dynamic_correctness','absolute','component_zvp','primary_reward'): qp=q
   elif a.arm=='cgh_noparallel': qp=q+'\n\nSolve step by step and verify the arithmetic. Answer in the original language of the question.'
   else: qp=old.PIVOT_TMPL.format(q=q,en=per['en'][i]['q'])
   extra,ept=generate(qp,a.extra); rewards(extra,lg,ex['gold'])
   weights=[]; log_exact=[]; log_root=[]
   for r in extra:
    with torch.no_grad():
     lr=float(lp(q,r)-lp(qp,r)); ratio=float(np.exp(np.clip(lr,np.log(.5),np.log(2.))))
    weights.append(ratio); log_root.append(lr); log_exact.append(lr*len(r['ids']))
   # Pure function shared with CPU checks; all terms score ORIGINAL q.
   from control_coefficients import build
   indexed,admitted=build('cgh_strict' if a.arm=='cgh_noparallel' else a.arm,y,aux(rs),vals(extra,'corr'),aux(extra),weights)
   both=rs+extra
   terms=[(both[int(j)],float(c)) for j,c in indexed]
   if a.arm=='component_zvp' and deg:
    for r,ax in zip(rs,norm(aux(rs))):
     scores,entropy=lp(q,r,True,token_entropy=True)
     target=entropy if r['c']['corr']>0 else -(entropy.max()-entropy)
     loss=-((float(ax)+a.entropy_alpha*target)*scores).mean()/8
     loss.backward()
   else:
    for r,c in terms:
     if float(c)!=0: (-float(c)/8*lp(q,r,True)).backward()
   row={'step':step+1,'batch':b,'lang':lg,'i':i,'degenerate':deg,'all_wrong':bool(y.sum()==0),'admitted':admitted,'generated_responses':4+a.extra,'generated_tokens':sum(len(r['ids']) for r in rs+extra),'prompt_tokens':pt+ept,'weights':weights,'log_exact_ratios':log_exact,'log_root_ratios':log_root,'extra_lengths':[len(r['ids']) for r in extra],'extra_target':vals(extra,'corr').tolist(),'scalar_zero':bool(np.std(y+aux(rs))<1e-12),'elapsed_s':time.time()-started}
   hist.append(row)
   with (out/'train.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
  opt.step()
  print(json.dumps({'step':step+1,'elapsed_min':(time.time()-started)/60,'peak_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)
  if (step+1)%50==0: model.save_pretrained(out/'adapter')
 model.save_pretrained(out/'adapter')
 result={'arm':a.arm,'seed':a.seed,'steps':a.steps,'eval_end':evaluate(),'total_generated_tokens':sum(r['generated_tokens'] for r in hist),'total_prompt_tokens':sum(r['prompt_tokens'] for r in hist),'elapsed_s':time.time()-started}
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print('DONE',out,flush=True)
if __name__=='__main__': main()
