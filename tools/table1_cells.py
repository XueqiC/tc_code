#!/usr/bin/env python3
"""Print Table 1 BFCL cells (mean±std over seeds, percent) from the standard-campaign CSVs
of the unified-budget runs. Usage: python3 tools/table1_cells.py [root] [cap]"""
import csv, glob, re, statistics, sys
from collections import defaultdict
root = sys.argv[1] if len(sys.argv) > 1 else '/home/xueqi/hq/projects/tc-alignment-table1'
cap = sys.argv[2] if len(sys.argv) > 2 else '13843'
NAMES = {'sft': 'Trajectory SFT', 'sad': 'Span-weighted SFT (SAD adapt.)', 'bbopd': 'On-policy teacher correction + SFT',
         'ddpo': 'SFT + teacher-ranked DPO (dDPO)', 'pbsd_insp': 'Offline teacher--student pref. (PBSD-insp.)',
         'pbsd_agent': 'PBSD --- agent adaptation', 'star': 'STaR w/o rationalization (0 teacher)'}
AX = [('Overall Acc', 'Overall'), ('Non-Live AST Acc', 'NL'), ('Live Acc', 'Live'), ('Multi Turn Acc', 'MT'), ('Memory Acc', 'Mem'), ('Irrelevance Detection', 'Irrel')]
cells = defaultdict(dict)
for p in sorted(glob.glob(f'{root}/results/bfcl_std/bfclB{cap}_*_s[0-9]/data_overall.csv')):
    m = re.search(rf'bfclB{cap}_(\w+)_s(\d)', p); r = list(csv.DictReader(open(p)))[0]
    cells[m.group(1)][int(m.group(2))] = {k: float(r[k].rstrip('%')) for k, _ in AX}
for arm in NAMES:
    v = cells.get(arm, {})
    if not v: print(f'{NAMES[arm]:48s} pending'); continue
    ov = [v[s]['Overall Acc'] for s in sorted(v)]
    sd = f'{statistics.stdev(ov):.1f}' if len(ov) > 1 else '--'
    axes = ' '.join(f"{lab} {statistics.mean(v[s][k] for s in v):.1f}" for k, lab in AX[1:])
    print(f'{NAMES[arm]:48s} seeds {sorted(v)} Overall {statistics.mean(ov):.1f}±{sd} ({"/".join(f"{x:.2f}" for x in ov)}) | {axes}')
