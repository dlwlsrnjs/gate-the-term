import numpy as np
from control_coefficients import build,normalize
x=[0,0,0,0]; aux=[0,.2,.4,.8]; ex=[1,0]; ea=[.9,.1]; w=[.7,1.1]
def arm(name,y=x):return build(name,y,aux,ex,ea,w)[0]
strict=arm('cgh_strict'); assert np.allclose([v for i,v in strict[:4]],normalize(aux))
assert strict[-1]==(4,.7) and len(strict)==5
assert not np.allclose([v for i,v in arm('cgh_legacy')[:4]],normalize(aux))
assert arm('cgh_strict_nois')==arm('pivot_rsft') # equality on this all-wrong group is intentional
assert arm('cgh_strict',[0,1,0,1])==arm('split',[0,1,0,1])
assert arm('cgh_strict',[1,1,1,1])==arm('split',[1,1,1,1])
assert arm('whole_gate')==[(4,.7)]
assert len(arm('q_resample'))==6
assert np.allclose([v for i,v in arm('absolute')],normalize(aux)-1)
assert np.isclose(sum(v for i,v in arm('vanilla')),0)
print('PASS: auxiliary preservation, legacy perturbation, RSFT identity, nondegenerate and all-correct branches, group gate, resampling, absolute signal.')

assert arm('dynamic_correctness') == [(4+j, float(v)*2) for j,v in enumerate(normalize(np.array(ea)+np.array(ex)))]
