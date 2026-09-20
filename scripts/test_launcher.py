"""CPU-only regression checks of complete/incomplete/config-mismatched job handling."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temp:
 out=Path(temp);job=out/'vanilla_s1_n100';job.mkdir()
 cmd=[sys.executable,str(ROOT/'scripts/run_suite.py'),'--arms','vanilla','--seeds','1','--model','/test/a09a35458c702b33eeacc393d103063234e8bc28','--output',str(out)]
 def run():return subprocess.run(cmd,capture_output=True,text=True)
 r=run();assert r.returncode!=0 and 'Incomplete run' in r.stderr
 c={'arm':'vanilla','seed':1,'steps':100,'eval_n':50,'model':'/test/a09a35458c702b33eeacc393d103063234e8bc28','eval_batch':8,'extra':2,'max_new':320,'entropy_alpha':1.,'source_sha256':hashlib.sha256((ROOT/'rebuttal_3213/train_controls.py').read_bytes()).hexdigest(),'coefficients_sha256':hashlib.sha256((ROOT/'rebuttal_3213/control_coefficients.py').read_bytes()).hexdigest()}
 (job/'config.json').write_text(json.dumps(c));d={'arm':'vanilla','seed':1,'steps':100,'eval_end':{lg:{'per_problem':[{}]*50} for lg in ('bn','te','th')}};(job/'result.json').write_text(json.dumps(d))
 r=run();assert r.returncode==0 and 'SKIP complete' in r.stdout,(r.stdout,r.stderr)
 c['extra']=3;(job/'config.json').write_text(json.dumps(c));r=run();assert r.returncode!=0 and 'Configuration mismatch' in r.stderr
print('PASS: incomplete refusal, validated completion skip, changed configuration refusal; no GPU used.')
