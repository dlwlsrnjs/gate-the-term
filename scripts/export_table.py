"""Generate a standalone LaTeX table from a validated completed cohort summary."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('--output',type=Path,default=Path('controls_table.tex'));a=p.parse_args()
d=json.loads(a.summary.read_text())
rows=[r'\begin{tabular}{lrr}',r'\hline',r'Arm & Accuracy (\%) & Script consistency (\%) \\',r'\hline']
for arm,v in d['arms'].items():rows.append(arm.replace('_',r'\_')+f" & {100*v['corr']:.2f} & {100*v['lang']:.2f}"+r' \\')
rows += [r'\hline',r'\end{tabular}',f"% Seeds: {d['seeds']}; {d['distinct_semantic_items']} semantic IDs; languages: {d['languages']}.",'% Response-count cap matched; not token/FLOP matched. Compare within this cohort only.']
a.output.write_text('\n'.join(rows)+'\n');print(a.output)
