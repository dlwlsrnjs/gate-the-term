"""Pure coefficient construction for all prospective controls; indices refer to
original responses first, followed by extra responses. No model imports needed.
"""
import numpy as np

def normalize(x):
 x=np.asarray(x,dtype=float)
 return (x-x.mean())/(x.std()+1e-4)

def build(arm, target, auxiliary, extra_target, extra_auxiliary, weights):
 y=np.asarray(target,dtype=float); ax=np.asarray(auxiliary,dtype=float)
 ey=np.asarray(extra_target,dtype=float); ea=np.asarray(extra_auxiliary,dtype=float)
 w=np.asarray(weights,dtype=float); g=len(y); k=len(ey)
 assert len(ax)==g and len(ea)==len(w)==k and k>0
 deg=bool(y.max()==y.min())
 split=normalize(ax)+normalize(y)
 coeff=normalize(ax+y) if arm in ('vanilla','cgh_legacy') else split
 terms=list(enumerate(coeff)); admitted=0
 if arm=='cgh_legacy' and deg:
  keep=np.flatnonzero(ey!=y[0]); yy=np.r_[y,ey[keep]]
  c=normalize(np.r_[ax,ea[keep]])
  if 0<yy.sum()<len(yy): c+=yy
  c*=np.r_[np.ones(g),w[keep]*2/k]
  terms=list(zip(list(range(g))+list(g+keep),c)); admitted=len(keep)
 elif arm in ('cgh_strict','cgh_strict_nois','whole_gate') and deg:
  if arm=='whole_gate': terms=[]
  if y[0]==0:
   for j,v in enumerate(ey):
    if v==1:
     terms.append((g+j,(w[j] if arm!='cgh_strict_nois' else 1.)*2/k)); admitted+=1
 elif arm=='pivot_rsft':
  for j,v in enumerate(ey):
   if v==1: terms.append((g+j,2/k)); admitted+=1
 elif arm=='q_resample' and deg:
  c=(normalize(np.r_[ax,ea])+normalize(np.r_[y,ey]))*g/(g+k)
  terms=list(enumerate(c)); admitted=k
 elif arm=='dynamic_correctness':
  # Budget-capped contrast filtering on two microgroups of size K (here K=2).
  # Keep the original G group if it has correctness contrast; otherwise use
  # the new K group only if it has contrast. Not the full DAPO algorithm.
  if deg:
   terms=[]
   if ey.max()!=ey.min():
    terms=[(g+j,float(c)*g/k) for j,c in enumerate(normalize(ea+ey))]
    admitted=k
  else: terms=list(enumerate(normalize(ax+y)))
 elif arm=='primary_reward':
  terms=list(enumerate(normalize(y*(1+ax))))
 elif arm=='absolute' and deg:
  terms=list(enumerate(normalize(ax)+(2*y-1)))
 return terms, admitted
